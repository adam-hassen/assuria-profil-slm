"""
Rapport de comparaison des petits modèles, à partir des résultats d'evaluer.py.

Lit resultats/<modele>/resume.json, par_champ.csv et reponses.jsonl pour chaque modèle évalué, et écrit :
  resultats/rapport_benchmark.html   rapport complet, à ouvrir dans le navigateur (ou à imprimer en PDF)
  resultats/rapport_benchmark.md     même contenu en Markdown, pour le rapport écrit

Lancement :  python rapport.py
"""
from __future__ import annotations

import csv
import html
import json
from datetime import date
from pathlib import Path

from schema import CHAMPS

DOSSIER = Path(__file__).parent / "resultats"
MESURES = [  # clé, libellé, plus haut = mieux ?, format
    ("json_valide", "JSON valide", True, "pct"),
    ("exactitude", "Exactitude", True, "pct"),
    ("oublis", "Oublis", False, "pct"),
    ("inventions", "Inventions", False, "pct"),
    ("profils_parfaits", "Profils parfaits", True, "pct"),
    ("latence_moyenne_s", "Latence moyenne", False, "s"),
    ("jetons_par_s", "Jetons par seconde", True, "n"),
]


def fmt(v, f):
    if v is None:
        return "—"
    return f"{v * 100:.1f} %".replace(".", ",") if f == "pct" else f"{v:.2f} s".replace(".", ",") if f == "s" else f"{v:.0f}"


def charger():
    modeles = []
    for d in sorted(DOSSIER.iterdir()) if DOSSIER.exists() else []:
        if not (d / "resume.json").exists():
            continue
        r = json.loads((d / "resume.json").read_text(encoding="utf-8"))
        with open(d / "par_champ.csv", encoding="utf-8") as f:
            r["champs"] = {l["champ"]: l for l in csv.DictReader(f, delimiter=";")}
        r["exemples"] = []
        if (d / "reponses.jsonl").exists():
            for ligne in open(d / "reponses.jsonl", encoding="utf-8"):
                e = json.loads(ligne)
                debut, fin = e["reponse"].find("{"), e["reponse"].rfind("}")
                try:
                    pred = json.loads(e["reponse"][debut:fin + 1])
                except (json.JSONDecodeError, ValueError):
                    pred = {}
                for c, att in e["attendu"].items():
                    p = pred.get(c)
                    faux = (att is None and p not in (None, "", "null")) or (att is not None and str(p).lower() != str(att).lower())
                    if faux and len(r["exemples"]) < 8 and c not in [x[0] for x in r["exemples"]]:
                        r["exemples"].append((c, att, p))
        modeles.append(r)
    return modeles


def constats(modeles):
    """Phrases de lecture automatiques : meilleur modèle par mesure, gain du fine-tuning."""
    lignes = []
    for cle, lib, haut, f in MESURES[:5]:
        vals = [(m[cle], m["nom"]) for m in modeles if m.get(cle) is not None]
        if len(vals) > 1:
            v, nom = (max if haut else min)(vals)
            if all(x == v for x, _ in vals):
                lignes.append(f"{lib} : égalité ({fmt(v, f)} pour tous les modèles).")
            else:
                lignes.append(f"Meilleur modèle en {lib.lower()} : {nom} ({fmt(v, f)}).")
    base = [m for m in modeles if "base" in m["nom"]]
    fine = [m for m in modeles if "fine" in m["nom"]]
    if base and fine:
        b, ft = base[0], fine[0]
        lignes.append(f"Fine-tuning ({b['nom']} → {ft['nom']}) : exactitude {fmt(b['exactitude'], 'pct')} → "
                      f"{fmt(ft['exactitude'], 'pct')}, inventions {fmt(b['inventions'], 'pct')} → {fmt(ft['inventions'], 'pct')}, "
                      f"profils parfaits {fmt(b['profils_parfaits'], 'pct')} → {fmt(ft['profils_parfaits'], 'pct')}.")
    n = {m["conversations"] for m in modeles}
    if len(n) > 1:
        lignes.append("Attention : les modèles n'ont pas été évalués sur le même nombre de conversations "
                      f"({', '.join(str(x) for x in sorted(n))}). Relancer avec le même --n pour une comparaison juste.")
    return lignes


def markdown(modeles):
    l = [f"# Benchmark du petit modèle : construction du profil", "", f"Généré le {date.today():%d/%m/%Y} par `rapport.py`.", "",
         "## Protocole", "",
         "Chaque modèle reçoit la même consigne (`schema.py`) et les mêmes conversations de test, jamais vues pendant "
         "l'entraînement, et doit rendre la fiche du client en JSON (28 champs). Chaque champ est comparé au profil attendu.", "",
         "## Résultats", "", "| Modèle | Conversations | " + " | ".join(x[1] for x in MESURES) + " |",
         "| --- | --- | " + " | ".join("---" for _ in MESURES) + " |"]
    for m in modeles:
        l.append(f"| {m['nom']} | {m['conversations']} | " + " | ".join(fmt(m.get(k), f) for k, _, _, f in MESURES) + " |")
    l += ["", "## Lecture", ""] + [f"- {x}" for x in constats(modeles)]
    l += ["", "## Exactitude par champ", "", "| Champ | " + " | ".join(m["nom"] for m in modeles) + " |",
          "| --- | " + " | ".join("---" for _ in modeles) + " |"]
    for c in CHAMPS:
        l.append(f"| `{c}` | " + " | ".join(fmt(float(m["champs"][c]["exactitude"]), "pct") if c in m["champs"] and
                                            int(m["champs"][c]["donnes"]) else "—" for m in modeles) + " |")
    l += ["", "## Exemples d'erreurs", ""]
    for m in modeles:
        l.append(f"**{m['nom']}**")
        l += [f"- `{c}` : attendu `{a}`, obtenu `{p}`" for c, a, p in m["exemples"]] or ["- aucune erreur relevée"]
        l.append("")
    return "\n".join(l)


def page(modeles):
    e = html.escape
    meilleurs = {}
    for cle, _, haut, _ in MESURES:
        vals = [m.get(cle) for m in modeles if m.get(cle) is not None]
        if vals:
            meilleurs[cle] = max(vals) if haut else min(vals)
    lignes = "".join(
        f"<tr><th>{e(m['nom'])}<small>{e(str(m['modele']))} · {m['conversations']} conversations</small></th>" +
        "".join(f"<td class='{'meilleur' if m.get(k) == meilleurs.get(k) and len(modeles) > 1 else ''}'>{fmt(m.get(k), f)}</td>"
                for k, _, _, f in MESURES) + "</tr>" for m in modeles)
    barres = ""
    for cle, lib, haut, f in MESURES[:5]:
        barres += f"<div class='mesure'><h3>{lib}<span>{'plus haut = mieux' if haut else 'plus bas = mieux'}</span></h3>"
        for m in modeles:
            v = m.get(cle) or 0
            barres += (f"<div class='barre'><span class='nom'>{e(m['nom'])}</span><span class='piste'>"
                       f"<span class='plein' style='width:{v * 100:.1f}%'></span></span><span class='val'>{fmt(v, f)}</span></div>")
        barres += "</div>"
    tete_champs = "".join(f"<th>{e(m['nom'])}</th>" for m in modeles)
    corps_champs = ""
    for c in CHAMPS:
        cases = ""
        for m in modeles:
            d = m["champs"].get(c)
            if d and int(d["donnes"]):
                v = float(d["exactitude"])
                teinte = "bon" if v >= .9 else "moyen" if v >= .7 else "faible"
                cases += f"<td class='{teinte}'>{fmt(v, 'pct')}</td>"
            else:
                cases += "<td>—</td>"
        corps_champs += f"<tr><th><code>{c}</code></th>{cases}</tr>"
    erreurs = "".join(f"<div class='erreurs'><h3>{e(m['nom'])}</h3><ul>" +
                      ("".join(f"<li><code>{c}</code> : attendu <b>{e(str(a))}</b>, obtenu <b>{e(str(p))}</b></li>" for c, a, p in m["exemples"])
                       or "<li>aucune erreur relevée</li>") + "</ul></div>" for m in modeles)
    lecture = "".join(f"<li>{e(x)}</li>" for x in constats(modeles)) or "<li>Évaluer au moins deux modèles pour obtenir une comparaison.</li>"
    entetes = "".join(f"<th>{lib}</th>" for _, lib, _, _ in MESURES)
    return f"""<!doctype html><html lang="fr"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Benchmark du petit modèle</title><style>
:root {{ --encre:#13232b; --doux:#5d6d75; --trait:#dde4e8; --accent:#0b5563; --clair:#e3f0f1; --fond:#f5f7f8; }}
body {{ margin:0; background:var(--fond); color:var(--encre); font:15px/1.55 "DM Sans",system-ui,sans-serif; }}
.page {{ max-width:1000px; margin:0 auto; padding:32px 20px 60px; }}
h1 {{ font-size:28px; margin:0 0 4px; letter-spacing:-.02em; }} h2 {{ font-size:19px; margin:34px 0 12px; }}
.sous {{ color:var(--doux); margin:0 0 20px; }}
.carte {{ background:#fff; border-radius:14px; padding:20px; box-shadow:0 1px 2px rgba(0,0,0,.05),0 6px 20px rgba(0,0,0,.05); overflow-x:auto; }}
table {{ border-collapse:collapse; width:100%; font-variant-numeric:tabular-nums; }}
th, td {{ padding:9px 10px; border-bottom:1px solid var(--trait); text-align:right; white-space:nowrap; }}
th:first-child {{ text-align:left; }} thead th {{ font-size:12.5px; color:var(--doux); font-weight:600; }}
tbody th small {{ display:block; color:var(--doux); font-weight:400; font-size:12px; }}
td.meilleur {{ color:var(--accent); font-weight:700; }}
td.bon {{ background:#e5f4ec; }} td.moyen {{ background:#fbf0dc; }} td.faible {{ background:#fbe3e0; }}
.mesures {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(280px,1fr)); gap:18px; }}
.mesure h3 {{ font-size:14px; margin:0 0 8px; display:flex; justify-content:space-between; }} .mesure h3 span {{ color:var(--doux); font-weight:400; font-size:12px; }}
.barre {{ display:grid; grid-template-columns:120px 1fr 64px; gap:8px; align-items:center; font-size:13px; margin:4px 0; }}
.nom {{ overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }} .val {{ text-align:right; font-variant-numeric:tabular-nums; }}
.piste {{ background:var(--clair); border-radius:4px; height:10px; }} .plein {{ display:block; height:100%; background:var(--accent); border-radius:4px; }}
ul {{ margin:6px 0 0; padding-left:20px; }} .erreurs h3 {{ font-size:14px; margin:14px 0 0; }}
code {{ font-size:13px; }} .note {{ color:var(--doux); font-size:13.5px; }}
@media print {{ body {{ background:#fff; }} .carte {{ box-shadow:none; border:1px solid var(--trait); }} }}
</style></head><body><div class="page">
<h1>Benchmark du petit modèle</h1>
<p class="sous">Construction du profil de l'assuré à partir de la conversation · généré le {date.today():%d/%m/%Y}</p>
<div class="carte"><p style="margin:0">Chaque modèle reçoit la même consigne et les mêmes conversations de test, jamais vues pendant
l'entraînement, et rend la fiche du client en JSON (28 champs). Chaque champ est comparé au profil attendu.
<b>Exactitude</b> : informations données bien remplies. <b>Oublis</b> : informations données laissées vides.
<b>Inventions</b> : champs jamais donnés remplis quand même. <b>Profils parfaits</b> : fiches entièrement justes.</p></div>
<h2>Résultats</h2><div class="carte"><table><thead><tr><th>Modèle</th>{entetes}</tr></thead><tbody>{lignes}</tbody></table></div>
<h2>Lecture</h2><div class="carte"><ul>{lecture}</ul></div>
<h2>Comparaison</h2><div class="carte mesures">{barres}</div>
<h2>Exactitude par champ</h2><div class="carte"><table><thead><tr><th>Champ</th>{tete_champs}</tr></thead><tbody>{corps_champs}</tbody></table>
<p class="note">Vert : 90 % ou plus. Orange : 70 à 90 %. Rouge : moins de 70 %.</p></div>
<h2>Exemples d'erreurs</h2><div class="carte">{erreurs}</div>
<p class="note">Limite : les conversations de test sont générées à partir de phrases types ; les résultats peuvent être meilleurs que
face à de vrais clients. La latence dépend de la machine : comparer les modèles sur la même machine.</p>
</div></body></html>"""


def main():
    modeles = charger()
    if not modeles:
        raise SystemExit("Aucun résultat dans resultats/ : lancer d'abord evaluer.py.")
    (DOSSIER / "rapport_benchmark.html").write_text(page(modeles), encoding="utf-8")
    (DOSSIER / "rapport_benchmark.md").write_text(markdown(modeles), encoding="utf-8")
    print(f"{len(modeles)} modèle(s) : {', '.join(m['nom'] for m in modeles)}")
    print(f"Rapport : {DOSSIER / 'rapport_benchmark.html'}")


if __name__ == "__main__":
    main()
