"""
Serveur de l'application AssurIA : la page web et l'API du conseiller.

Lancement (depuis la racine du dépôt) :
    uvicorn app.serveur:app --port 8000
puis ouvrir http://localhost:8000   (ajouter ?demo pour voir ce que fait le petit modèle à chaque tour)

Réglages dans le fichier .env à la racine (jamais envoyé sur GitHub) :
    NEON_DATABASE_URL=postgresql://...   base des offres
    SLM_MODEL=qwen2.5:1.5b               petit modèle qui remplit la fiche (plus tard : le modèle fine-tuné)
    LLM_MODEL=qwen3:8b                   gros LLM qui discute avec le client
"""
from __future__ import annotations

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

SLM = os.environ.get("SLM_MODEL", "qwen2.5:1.5b")
LLM = os.environ.get("LLM_MODEL", "qwen3:8b")
STATIQUE = Path(__file__).resolve().parent / "static"

app = FastAPI(title="AssurIA")
app.mount("/static", StaticFiles(directory=STATIQUE), name="static")
conversations: dict[str, agent.Conversation] = {}


def etat(c: agent.Conversation) -> dict:
    n, total = c.progression()
    return {"fiche": c.fiche, "connues": n, "total": total, "demandee": c.demandee}


@app.get("/")
def accueil():
    return FileResponse(STATIQUE / "index.html")


@app.post("/api/conversation")
def nouvelle():
    cid = uuid.uuid4().hex
    conversations[cid] = agent.Conversation(SLM, LLM)
    return {"id": cid, "message": agent.ACCUEIL, **etat(conversations[cid]), "modeles": {"petit": SLM, "gros": LLM},
            "donnees": offres.source()}


class Message(BaseModel):
    id: str
    texte: str


@app.post("/api/message")
def message(m: Message):
    c = conversations.get(m.id)
    if c is None:
        raise HTTPException(404, "Conversation inconnue : rechargez la page.")
    texte = m.texte.strip()[:1000]

    def flux():
        ligne = lambda d: json.dumps(d, ensure_ascii=False, default=str) + "\n"
        if c.terminee:
            yield ligne({"t": "message", "v": "Votre recherche est terminée. Cliquez sur « Nouvelle simulation » pour recommencer."})
            yield ligne({"t": "fin"})
            return
        suite = c.tour(texte, offres.deduire)
        yield ligne({"t": "fiche", **etat(c)})
        yield ligne({"t": "demo", **c.journal[-1]})
        if suite["suivante"]:
            q = c.question(suite["suivante"], suite["relance"], texte)
            c.historique.append({"role": "assistant", "content": q})
            yield ligne({"t": "message", "v": q})
        else:
            c.terminee = True
            try:
                trouvees, coefs, info = offres.chercher(c.fiche)
            except Exception as err:
                yield ligne({"t": "erreur", "v": f"Impossible de lire les offres ({err.__class__.__name__}). Vérifiez la connexion à la base."})
                yield ligne({"t": "fin"})
                return
            yield ligne({"t": "offres", "offres": trouvees, "coefs": coefs, "info": info})
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

    return StreamingResponse(flux(), media_type="application/x-ndjson")
