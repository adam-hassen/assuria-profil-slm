"""
Serveur de l'application AssurIA : la page web et l'API du conseiller.

Lancement (depuis la racine du dépôt) :
    uvicorn app.serveur:app --port 8000
puis ouvrir http://localhost:8000   (ajouter ?demo pour voir ce que fait le petit modèle à chaque tour)

Réglages dans le fichier .env à la racine (jamais envoyé sur GitHub) :
    NEON_DATABASE_URL=postgresql://...   base des offres
    SLM_MODEL=qwen2.5:1.5b               petit modèle qui remplit la fiche (plus tard : le modèle fine-tuné)
    LLM_MODEL=qwen3:8b                   gros LLM qui discute avec le client
    REGCHECK_USER=...                    compte RegCheck pour retrouver le véhicule par sa plaque (facultatif)
"""
from __future__ import annotations

import copy
import json
import os
import sys
import uuid
from pathlib import Path

RACINE = Path(__file__).resolve().parent.parent
env = RACINE / ".env"
if env.exists():
    for ligne in env.read_text().splitlines():
        if "=" in ligne and not ligne.strip().startswith("#"):
            k, v = ligne.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"'))

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fastapi import FastAPI, HTTPException  # noqa: E402
from fastapi.responses import FileResponse, StreamingResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402
from pydantic import BaseModel  # noqa: E402

import agent  # noqa: E402
import offres  # noqa: E402
import plaque  # noqa: E402

SLM = os.environ.get("SLM_MODEL", "qwen2.5:1.5b")
LLM = os.environ.get("LLM_MODEL", "qwen3:8b")
STATIQUE = Path(__file__).resolve().parent / "static"

app = FastAPI(title="AssurIA")
app.mount("/static", StaticFiles(directory=STATIQUE), name="static")
conversations: dict[str, agent.Conversation] = {}


def etat(c: agent.Conversation) -> dict:
    n, total = c.progression()
    return {"fiche": c.fiche, "connues": n, "total": total, "demandee": c.demandee, "ignorees": sorted(c.ignorees)}


@app.get("/")
def accueil():
    return FileResponse(STATIQUE / "index.html")


@app.post("/api/conversation")
def nouvelle():
    cid = uuid.uuid4().hex
    conversations[cid] = agent.Conversation(SLM, LLM)
    return {"id": cid, "message": agent.ACCUEIL, **etat(conversations[cid]), "modeles": {"petit": SLM, "gros": LLM},
            "donnees": offres.source(), "plaque": plaque.disponible()}


class Message(BaseModel):
    id: str
    texte: str


def suite_de_conversation(c: agent.Conversation, suite: dict, texte: str, avant=()):
    """Flux NDJSON envoyé à la page : fiche, coulisses, puis la question suivante ou les offres."""
    ligne = lambda d: json.dumps(d, ensure_ascii=False, default=str) + "\n"
    yield ligne({"t": "fiche", **etat(c)})
    yield ligne({"t": "demo", **c.journal[-1]})
    for e in avant:
        yield ligne(e)
    if suite["suivante"]:
        q = c.question(suite["suivante"], suite["relance"], texte, suite.get("passee"))
        c.historique.append({"role": "assistant", "content": q})
        yield ligne({"t": "message", "v": q})
    elif c.critere is None:
        # profil complet : on demande au client ce qui compte le plus avant de présenter les offres
        c.demandee = "critere"
        c.historique.append({"role": "assistant", "content": agent.QUESTION_CRITERE})
        yield ligne({"t": "fiche", **etat(c)})
        yield ligne({"t": "message", "v": agent.QUESTION_CRITERE})
    else:
        c.terminee = True
        c.demandee = None
        try:
            toutes, coefs, info = offres.chercher(c.fiche)
            par_critere = {k: offres.classer(copy.deepcopy(toutes), k)[:5] for k in agent.CRITERES}
            trouvees = par_critere[c.critere]
        except Exception as err:
            yield ligne({"t": "erreur", "v": f"Impossible de lire les offres ({err.__class__.__name__}). Vérifiez la connexion à la base."})
            yield ligne({"t": "fin"})
            return
        libelles = {"age": "âge", "commune": "ville", "pays_residence": "pays", "annees_permis": "ancienneté du permis",
                    "bonus_malus": "bonus-malus", "kilometrage_annuel": "kilométrage", "usage": "usage",
                    "stationnement_nuit": "stationnement", "formule_souhaitee": "formule", "budget_mensuel_max": "budget",
                    "sinistres_responsables_36_mois": "sinistres", "deja_assure": "antécédents", "type_permis": "permis",
                    "marque_modele": "modèle"}
        manquantes = [libelles.get(x, x) for x in sorted(c.ignorees)]
        yield ligne({"t": "fiche", **etat(c)})
        yield ligne({"t": "offres", "offres": trouvees, "coefs": coefs, "info": info if trouvees else "",
                     "critere": c.critere, "par_critere": par_critere, "manquantes": manquantes,
                     "comparees": len(toutes)})
        if trouvees:
            complet = ""
            try:
                for morceau in c.recommandation(trouvees):
                    complet += morceau
                    yield ligne({"t": "jeton", "v": morceau})
            except Exception:
                complet = (f"Je vous recommande {trouvees[0]['assureur']} ({trouvees[0]['produit']}), estimé à "
                           f"{trouvees[0]['prix_mensuel_estime_eur']} € par mois. Ces prix sont des estimations.")
                yield ligne({"t": "message", "v": complet})
            c.historique.append({"role": "assistant", "content": complet})
        else:
            yield ligne({"t": "message", "v": info or "Je n'ai trouvé aucune offre correspondante dans notre base."})
    yield ligne({"t": "fin"})


def terminee():
    yield json.dumps({"t": "message", "v": "Votre recherche est terminée. Cliquez sur « Nouvelle simulation » pour recommencer."},
                     ensure_ascii=False) + "\n"
    yield json.dumps({"t": "fin"}) + "\n"


def conversation(cid: str) -> agent.Conversation:
    c = conversations.get(cid)
    if c is None:
        raise HTTPException(404, "Conversation inconnue : rechargez la page.")
    return c


@app.post("/api/message")
def message(m: Message):
    c = conversation(m.id)
    texte = m.texte.strip()[:1000]
    if c.terminee:
        return StreamingResponse(terminee(), media_type="application/x-ndjson")
    if c.demandee == "critere":
        c.choisir_critere(texte)
        suite = {"suivante": None, "relance": False}
        c.journal.append({"petit_modele": "(choix du critère)", "duree_s": 0, "reponse_brute": "", "retenu_du_petit_modele": {},
                          "lu_dans_la_reponse": {"critere": c.critere}, "erreur": None})
    else:
        suite = c.tour(texte, offres.deduire)
    return StreamingResponse(suite_de_conversation(c, suite, texte), media_type="application/x-ndjson")


@app.post("/api/plaque")
def recherche_plaque(m: Message):
    """Le client a saisi sa plaque : on retrouve le véhicule et on passe à la question suivante."""
    c = conversation(m.id)
    if c.terminee:
        return StreamingResponse(terminee(), media_type="application/x-ndjson")
    p = plaque.trouver_plaque(m.texte) or m.texte.strip().upper()[:12]
    resultat = plaque.chercher(p, c.fiche.get("type_vehicule"))
    suite = c.vehicule_par_plaque(p, resultat)
    if resultat["ok"]:
        avant = [{"t": "vehicule", "plaque": p, **resultat["vehicule"]}]
        texte = f"Mon véhicule est une {resultat['vehicule']['nom']}."
        return StreamingResponse(suite_de_conversation(c, suite, texte, avant), media_type="application/x-ndjson")
    v = ("Je n'ai pas retrouvé de véhicule avec cette plaque. Vérifiez-la et réessayez, "
         "ou indiquez-moi directement la marque et le modèle.")
    c.historique.append({"role": "assistant", "content": v})
    flux = (json.dumps(d, ensure_ascii=False, default=str) + "\n"
            for d in ({"t": "fiche", **etat(c)}, {"t": "demo", **c.journal[-1]}, {"t": "message", "v": v}, {"t": "fin"}))
    return StreamingResponse(flux, media_type="application/x-ndjson")
