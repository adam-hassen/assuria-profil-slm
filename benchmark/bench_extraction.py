"""Test 2 : extraction du profil sur les 30 conversations écrites à la main.

Chaque approche reçoit la conversation complète et renvoie une fiche :
  A. petit modèle fine-tuné d'Adam      -> réponses déjà calculées par evaluer.py (resultats/finetune-ecrit)
  B. agent « profile-extractor » d'Ismail (qwen3:8b, sa consigne, son schéma) -> appelé via le pont Mastra
On note les 12 champs que les deux fiches ont en commun (la franchise n'existe pas dans le schéma d'extraction d'Ismail).

    python3 benchmark/bench_extraction.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ICI = Path(__file__).parent
sys.path.insert(0, str(ICI))
import client_simule as cs  # noqa: E402
from bench_conversations import post  # noqa: E402

RACINE = ICI.parent
CHAMPS = [c for c in cs.COMMUNS if c != "franchise"]


def lire(t: str) -> dict:
    try:
        return json.loads(t[t.find("{"): t.rfind("}") + 1])
    except Exception:
        return {}


def noter(attendu: dict, obtenu: dict) -> dict:
    v = cs.depuis_adam(attendu)
    dits = {c for c in CHAMPS if v[c] is not None}
    return {c: n for c, n in cs.noter({**v}, obtenu, dits).items() if c in CHAMPS}


def resumer(nom: str, lignes: list[dict]) -> dict:
    notes = [v for l in lignes for v in l["notes"].values()]
    dits = sum(v in ("juste", "faux", "oublie") for v in notes)
    non_dits = sum(v in ("invente", "vide") for v in notes)
    durees = [l["duree_s"] for l in lignes if l.get("duree_s")]
    return {"nom": nom, "conversations": len(lignes),
            "exactitude": round(sum(v == "juste" for v in notes) / dits, 4),
            "oublis": round(sum(v == "oublie" for v in notes) / dits, 4),
            "faux": sum(v == "faux" for v in notes),
            "inventions": round(sum(v == "invente" for v in notes) / non_dits, 4) if non_dits else 0.0,
            "fiches_parfaites": sum(all(v in ("juste", "vide") for v in l["notes"].values()) for l in lignes),
            "latence_moyenne_s": round(sum(durees) / len(durees), 2) if durees else None,
            "par_champ": {c: {k: sum(l["notes"][c] == k for l in lignes) for k in ("juste", "faux", "oublie", "invente", "vide")} for c in CHAMPS}}


def main() -> int:
    convs = {d["id_profil"]: d for d in map(json.loads, open(RACINE / "donnees/test_ecrit_conversations.jsonl"))}
    sortie = ICI / "resultats"
    sortie.mkdir(exist_ok=True)
    resumes = []
    # A (et les autres modèles déjà passés dans evaluer.py) : on relit leurs réponses
    for nom in ["base-ecrit", "finetune-ecrit", "qwen3-8b-ecrit"]:
        f = RACINE / "resultats" / nom / "reponses.jsonl"
        if not f.exists():
            continue
        lignes = []
        for d in map(json.loads, open(f)):
            obtenu = cs.depuis_adam(lire(d["reponse"]))
            lignes.append({"id": d["id_profil"], "notes": noter(d["attendu"], obtenu), "duree_s": d.get("latence_s")})
        r = resumer(nom, lignes)
        r["latence_moyenne_s"] = json.loads((f.parent / "resume.json").read_text()).get("latence_moyenne_s")
        resumes.append(r)
    # B : l'extracteur d'Ismail
    lignes = []
    with open(sortie / "extraction_ismail.jsonl", "w", encoding="utf-8") as f:
        for i, (k, d) in enumerate(sorted(convs.items()), 1):
            texte = "\n".join(f"{qui} : {t}" for qui, t in d["tours"])
            t0 = time.perf_counter()
            try:
                r = post("http://localhost:4200/extraire", {"text": texte})
                profil, err = r.get("profile") or {}, None
            except Exception as e:
                profil, err = {}, str(e)
            duree = round(time.perf_counter() - t0, 2)
            notes = noter(d["profil_attendu"], cs.depuis_ismail(profil))
            ligne = {"id": k, "profil": profil, "notes": notes, "duree_s": duree, "erreur": err}
            f.write(json.dumps(ligne, ensure_ascii=False) + "\n")
            lignes.append(ligne)
            print(f"  {i}/30 {k}  {duree:>5} s  justes {sum(v == 'juste' for v in notes.values())}  inventés {sum(v == 'invente' for v in notes.values())}{'  ERREUR ' + err if err else ''}")
    resumes.append(resumer("ismail-extracteur", lignes))
    (sortie / "resume_extraction.json").write_text(json.dumps(resumes, ensure_ascii=False, indent=2))
    for r in resumes:
        print(f"{r['nom']:<20} exactitude {r['exactitude']:.1%}  inventions {r['inventions']:.1%}  parfaites {r['fiches_parfaites']}/30  {r['latence_moyenne_s']} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
