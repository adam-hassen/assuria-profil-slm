"""Test 1 : conversations complètes avec un client simulé, sur les deux approches.

Le même client (même profil, mêmes phrases) parle à :
  A. l'application d'Adam   (FastAPI, http://localhost:8000)  -> petit modèle fine-tuné + code + qwen3:8b
  B. l'agent d'Ismail       (pont Mastra, http://localhost:4200) -> qwen3:8b agent avec outils

On mesure : conversation menée jusqu'aux offres ou non, nombre de tours, temps, et la qualité de la fiche finale
sur les 13 champs que les deux fiches ont en commun.

    python3 benchmark/bench_conversations.py --systeme adam   --n 10
    python3 benchmark/bench_conversations.py --systeme ismail --n 10
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

import pandas as pd

ICI = Path(__file__).parent
sys.path.insert(0, str(ICI))
import client_simule as cs  # noqa: E402

PROFILS = ICI.parent.parent / "assuria-projet" / "data" / "objectif1" / "profils" / "profils.parquet"
SORTIE = ICI / "resultats"
PREMIER = "Bonjour, je cherche une assurance pour ma voiture."
MAX_TOURS = 40

# champ de l'application d'Adam -> concept du client
CHAMP_ADAM = {"type_vehicule": "type_vehicule", "marque_modele": "vehicule", "annee_mise_en_circulation": "annee_vehicule",
              "energie": "energie", "mode_acquisition": "acquisition", "age": "age", "commune": "commune",
              "pays_residence": "pays", "type_permis": "type_permis", "annees_permis": "annees_permis",
              "pays_permis": "pays_permis", "deja_assure": "deja_assure", "conduite_accompagnee": "conduite_accompagnee",
              "bonus_malus": "bonus_malus", "sinistres_responsables_36_mois": "sinistres",
              "resiliation_assureur": "resiliation", "conducteur_secondaire": "conducteur_secondaire", "usage": "usage",
              "kilometrage_annuel": "km", "stationnement_nuit": "stationnement", "formule_souhaitee": "formule",
              "franchise_souhaitee": "franchise", "budget_mensuel_max": "budget", "critere": "critere"}
# champ de l'agent d'Ismail -> concept du client
CHAMP_ISMAIL = {"age": "age", "licenseYears": "annees_permis", "accompaniedDriving": "conduite_accompagnee",
                "licenceOrigin": "pays_permis", "bonusMalus": "bonus_malus", "claimFreeYears": "annees_sans_sinistre",
                "atFaultClaims": "sinistres", "vehicle": "vehicule", "vehicleYear": "annee_vehicule",
                "vehiclePower": "puissance", "vehicleEnergy": "energie", "vehicleNewPrice": "prix_neuf", "usage": "usage",
                "annualKm": "km", "parking": "stationnement", "city": "commune", "zoneType": "type_zone",
                "deductible": "franchise", "priorities": "garanties", "coverageLevel": "formule",
                "maxMonthlyBudget": "budget"}


def post(url: str, corps: dict, flux=False, timeout=900):
    req = urllib.request.Request(url, data=json.dumps(corps).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        brut = r.read().decode()
    return [json.loads(l) for l in brut.splitlines() if l.strip()] if flux else json.loads(brut)


# --------------------------------------------------------------------------- approche A
def conversation_adam(p, url: str) -> dict:
    d = post(f"{url}/api/conversation", {})
    cid, demandee, fiche = d["id"], d.get("demandee"), d.get("fiche", {})
    dits, journal, offres, fin = {"type_vehicule"}, [], [], None
    message, concept = PREMIER, "type_vehicule"
    for tour in range(1, MAX_TOURS + 1):
        t0 = time.perf_counter()
        evts = post(f"{url}/api/message", {"id": cid, "texte": message}, flux=True)
        duree = time.perf_counter() - t0
        texte = ""
        for e in evts:
            if e.get("t") == "fiche":
                fiche, demandee = e.get("fiche", fiche), e.get("demandee")
            elif e.get("t") == "message":
                texte = e["v"]
            elif e.get("t") == "jeton":
                texte += e["v"]
            elif e.get("t") == "offres":
                offres = e.get("offres", [])
                fin = "offres" if offres else "aucune offre"
            elif e.get("t") == "erreur":
                fin = "erreur : " + e["v"]
        journal.append({"tour": tour, "client": message, "concept": concept, "systeme": texte, "duree_s": round(duree, 2)})
        if fin or demandee is None:
            fin = fin or "terminée sans offres"
            break
        concept = CHAMP_ADAM.get(demandee)
        if concept is None:
            message, concept = "Je ne suis pas sûr de comprendre.", None
        else:
            message = cs.repondre(concept, p)
            dits.add(concept)
    else:
        fin = "trop de tours"
    return {"fin": fin, "fiche": fiche, "commun": cs.depuis_adam(fiche), "dits": sorted(dits), "journal": journal,
            "offres": [{"assureur": o["assureur"], "produit": o["produit"], "prix": o["prix_mensuel_estime_eur"]} for o in offres[:3]]}


# --------------------------------------------------------------------------- approche B
def conversation_ismail(p, url: str, numero: int) -> dict:
    fil = f"bench-{int(time.time())}-{numero}"
    dits, journal, manquants, fin, offres, profil = {"type_vehicule"}, [], [], None, [], {}
    message, concepts = PREMIER, ["type_vehicule"]
    sans_avancer = 0
    for tour in range(1, MAX_TOURS + 1):
        t0 = time.perf_counter()
        try:
            r = post(f"{url}/tour", {"thread": fil, "message": message})
        except Exception as err:  # le pont renvoie une erreur si l'agent plante
            journal.append({"tour": tour, "client": message, "concepts": concepts, "erreur": str(err)})
            fin = f"erreur : {err}"
            break
        duree = time.perf_counter() - t0
        avant = dict(profil)
        profil = r.get("profile") or profil
        if r.get("missing") is not None:
            manquants = [m["field"] for m in r["missing"]]
        journal.append({"tour": tour, "client": message, "concepts": concepts, "systeme": r.get("text", ""),
                        "outils": r.get("tools", []), "duree_s": round(duree, 2)})
        s = r.get("search")
        if s and s.get("status") == "ok":
            offres = s.get("offers", [])
            fin = "refus" if s.get("refusal") else ("offres" if offres else "aucune offre")
            break
        sans_avancer = sans_avancer + 1 if profil == avant else 0
        if sans_avancer >= 5:
            fin = "bloquée (5 tours sans avancer)"
            break
        texte = r.get("text", "")
        candidats = [CHAMP_ISMAIL[f] for f in manquants if f in CHAMP_ISMAIL] or list(CHAMP_ISMAIL.values())
        concepts = cs.concepts_demandes(texte, candidats)[:3]
        if not concepts and ("?" not in texte or cs.est_confirmation(texte)):
            message, concepts = "Oui, allez-y, cherchez-moi les offres.", []
        elif not concepts:
            concepts = candidats[:1]
            message = cs.repondre(concepts[0], p)
        else:
            message = " ".join(cs.repondre(c, p) for c in concepts)
        dits.update(concepts)
    else:
        fin = "trop de tours"
    return {"fin": fin, "fiche": profil, "commun": cs.depuis_ismail(profil), "dits": sorted(dits), "journal": journal,
            "offres": [{"assureur": o.get("insurer"), "produit": o.get("name"), "prix": o.get("monthlyPremium")} for o in offres[:3]]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--systeme", choices=["adam", "ismail"], required=True)
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--profils", default=str(PROFILS))
    ap.add_argument("--url", default=None)
    ap.add_argument("--graine", type=int, default=2026)
    a = ap.parse_args()
    url = a.url or ("http://localhost:8000" if a.systeme == "adam" else "http://localhost:4200")
    try:
        df = pd.read_parquet(a.profils)
    except (ImportError, FileNotFoundError, ValueError):  # sans pyarrow : la version CSV du même fichier
        df = pd.read_csv(Path(a.profils).with_suffix(".csv"), low_memory=False)
    # l'agent d'Ismail ne couvre que la voiture en France : on prend ces profils pour les deux approches
    df = df[(df.pays_residence == "France") & (df.type_vehicule == "auto") & (df.age >= 18) & (df.pays_permis == "France")]
    tirage = df.sample(a.n, random_state=a.graine)
    SORTIE.mkdir(exist_ok=True)
    fichier = SORTIE / f"conversations_{a.systeme}.jsonl"
    lignes = []
    with open(fichier, "w", encoding="utf-8") as f:
        for i, p in enumerate(tirage.itertuples(), 1):
            t0 = time.perf_counter()
            r = conversation_adam(p, url) if a.systeme == "adam" else conversation_ismail(p, url, i)
            r["duree_totale_s"] = round(time.perf_counter() - t0, 1)
            r["id_profil"] = p.id_profil
            r["notes"] = cs.noter(cs.verite(p), r["commun"], set(r["dits"]))
            r["verite"] = cs.verite(p)
            f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
            f.flush()
            lignes.append(r)
            n = r["notes"]
            print(f"  {i}/{a.n}  {r['fin']:<22} {len(r['journal']):>2} tours  {r['duree_totale_s']:>6} s  "
                  f"justes {sum(v == 'juste' for v in n.values())}/{sum(v in ('juste', 'faux', 'oublie') for v in n.values())}  "
                  f"inventés {sum(v == 'invente' for v in n.values())}")
    resume = resumer(lignes, a.systeme)
    (SORTIE / f"resume_conversations_{a.systeme}.json").write_text(json.dumps(resume, ensure_ascii=False, indent=2))
    print(json.dumps(resume, ensure_ascii=False, indent=2))
    return 0


def resumer(lignes: list[dict], systeme: str) -> dict:
    notes = [v for r in lignes for v in r["notes"].values()]
    dits = sum(v in ("juste", "faux", "oublie") for v in notes)
    non_dits = sum(v in ("invente", "vide") for v in notes)
    tours = [t["duree_s"] for r in lignes for t in r["journal"] if "duree_s" in t]
    reussies = [r for r in lignes if r["fin"] in ("offres", "refus", "aucune offre")]
    return {
        "systeme": systeme, "conversations": len(lignes),
        "menees_au_bout": len(reussies), "taux_menees_au_bout": round(len(reussies) / len(lignes), 3),
        "fins": {k: sum(r["fin"] == k for r in lignes) for k in sorted({r["fin"] for r in lignes})},
        "tours_moyens": round(sum(len(r["journal"]) for r in lignes) / len(lignes), 1),
        "duree_moyenne_conversation_s": round(sum(r["duree_totale_s"] for r in lignes) / len(lignes), 1),
        "duree_moyenne_tour_s": round(sum(tours) / len(tours), 2) if tours else None,
        "duree_max_tour_s": round(max(tours), 2) if tours else None,
        "champs_dits": dits, "exactitude": round(sum(v == "juste" for v in notes) / dits, 4) if dits else None,
        "faux": sum(v == "faux" for v in notes), "oublies": sum(v == "oublie" for v in notes),
        "inventions": round(sum(v == "invente" for v in notes) / non_dits, 4) if non_dits else 0.0,
        "inventes": sum(v == "invente" for v in notes),
        "fiches_parfaites": sum(all(v in ("juste", "vide") for v in r["notes"].values()) for r in lignes),
        "par_champ": {c: {k: sum(r["notes"][c] == k for r in lignes) for k in ("juste", "faux", "oublie", "invente", "vide")}
                      for c in cs.COMMUNS},
    }


if __name__ == "__main__":
    sys.exit(main())
