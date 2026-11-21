"""
Le cerveau de l'application : le petit modèle remplit la fiche, le gros LLM discute avec le client.

À chaque message du client :
  1. le PETIT modèle (Ollama) lit toute la conversation et renvoie la fiche JSON (consigne de schema.py) ;
  2. le code nettoie la fiche (valeurs autorisées seulement) et regarde quelles cases importantes sont vides ;
  3. le GROS LLM (Ollama) reçoit la fiche et la case à demander, et pose la question suivante de façon naturelle ;
  4. quand la fiche est complète, le code cherche les offres (offres.py) et le gros LLM explique la recommandation.
"""
from __future__ import annotations

import json
import re
import sys
import time
import unicodedata
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from schema import CHAMPS, CONSIGNE, NOMS, message_utilisateur  # noqa: E402

OLLAMA = "http://localhost:11434"

# Cases nécessaires pour chercher les offres, dans l'ordre où on les demande, avec ce qu'il faut demander
A_DEMANDER = [
    ("type_vehicule", "s'il veut assurer une voiture ou une moto"),
    ("marque_modele", "la marque et le modèle du véhicule"),
    ("age", "son âge"),
    ("commune", "la ville où il habite"),
    ("pays_residence", "le pays où il habite"),
    ("type_permis", "le type de permis qu'il a"),
    ("annees_permis", "depuis combien de temps il a son permis"),
    ("deja_assure", "s'il a déjà été assuré comme conducteur principal"),
    ("bonus_malus", "son bonus-malus (indiqué sur son relevé d'information)"),
    ("sinistres_responsables_36_mois", "combien de sinistres il a eus ces 3 dernières années, et s'il était responsable"),
    ("usage", "l'usage du véhicule : privé, trajet travail ou professionnel"),
    ("kilometrage_annuel", "combien de kilomètres il fait par an"),
    ("stationnement_nuit", "où le véhicule est garé la nuit : garage fermé, parking privé ou dans la rue"),
    ("formule_souhaitee", "la formule souhaitée : au tiers, intermédiaire ou tous risques"),
    ("budget_mensuel_max", "son budget maximum par mois"),
]

ACCUEIL = ("Bonjour ! Je suis votre conseiller en assurance auto et moto. Je vais vous poser quelques questions "
           "pour trouver les offres qui vous correspondent. Pour commencer, que souhaitez-vous assurer ?")

CONSIGNE_CONSEILLER = """Tu es un conseiller en assurance auto et moto, chaleureux et professionnel. Tu parles français.
Tu aides le client à faire un devis en lui posant des questions, une seule à la fois.
Règles :
- Réponds en 1 ou 2 phrases courtes.
- Si le client a posé une question, réponds-y brièvement d'abord.
- Pose ensuite UNE seule question, sur l'information demandée ci-dessous. Ne redemande jamais une information déjà connue.
- Ne cite jamais de prix, d'assureur ni d'offre : ils seront calculés plus tard.
- N'écris pas la fiche, pas de JSON, pas de liste."""

CONSIGNE_RECOMMANDATION = """Tu es un conseiller en assurance auto et moto. Tu parles français.
On te donne la fiche du client et les offres trouvées dans notre base, déjà classées par prix estimé.
Présente les 3 premières offres en quelques phrases : assureur, produit, prix estimé par mois, et pourquoi elle correspond
au client (budget, formule, garanties). Recommande la première, sauf si une autre correspond clairement mieux.
Règles : n'utilise QUE les offres et les chiffres donnés, n'invente aucun prix ni garantie. Précise que les prix sont des
estimations et qu'un devis auprès de l'assureur est nécessaire. Sois concis (8 lignes maximum)."""


# --------------------------------------------------------------------------- appels à Ollama
def ollama_chat(modele: str, messages: list[dict], json_mode=False, stream=False, options=None):
    corps = {"model": modele, "messages": messages, "stream": stream, "think": False,
             "options": {"temperature": 0, "num_ctx": 4096, **(options or {})}}
    if json_mode:
        corps["format"] = "json"
    req = urllib.request.Request(f"{OLLAMA}/api/chat", data=json.dumps(corps).encode(),
                                 headers={"Content-Type": "application/json"})
    rep = urllib.request.urlopen(req, timeout=600)
    if not stream:
        return json.loads(rep.read())["message"]["content"]

    def morceaux():
        for ligne in rep:
            if ligne.strip():
                d = json.loads(ligne)
                yield d.get("message", {}).get("content", "")
    return morceaux()


def modeles_installes() -> list[str]:
    try:
        with urllib.request.urlopen(f"{OLLAMA}/api/tags", timeout=5) as r:
            return sorted(m["name"] for m in json.loads(r.read())["models"])
    except Exception:
        return []


# --------------------------------------------------------------------------- nettoyage de la fiche
def _simple(v) -> str:
    s = unicodedata.normalize("NFKD", str(v)).encode("ascii", "ignore").decode().lower().strip()
    return re.sub(r"[\s\-]+", "_", s)


VALEURS = {c: [v.strip() for v in d.split("|")] for c, (t, d) in CHAMPS.items() if t == "texte" and "|" in d}


def nettoyer(brut: dict) -> dict:
    """Garde seulement les valeurs valides : une valeur hors liste ou illisible devient null."""
    propre = {}
    for c in NOMS:
        v = brut.get(c)
        if v is None or (isinstance(v, str) and _simple(v) in ("", "null", "none", "inconnu", "n/a")):
            propre[c] = None
            continue
        t = CHAMPS[c][0]
        try:
            if t == "entier":
                propre[c] = int(float(str(v).replace(" ", "").replace(",", ".")))
            elif t == "booléen":
                propre[c] = v if isinstance(v, bool) else _simple(v) in ("true", "oui", "vrai", "1")
            elif c in VALEURS:
                ok = {_simple(x): x for x in VALEURS[c]}
                propre[c] = ok.get(_simple(v))
            else:
                propre[c] = str(v).strip()
        except (ValueError, TypeError):
            propre[c] = None
    return propre


def lire_json(texte: str) -> dict:
    debut, fin = texte.find("{"), texte.rfind("}")
    try:
        d = json.loads(texte[debut:fin + 1])
        return d if isinstance(d, dict) else {}
    except (json.JSONDecodeError, ValueError):
        return {}


# --------------------------------------------------------------------------- l'agent
def texte_conversation(historique: list[dict]) -> str:
    return "\n".join(f"{'Conseiller' if m['role'] == 'assistant' else 'Client'} : {m['content']}" for m in historique)


def extraire_fiche(modele_slm: str, historique: list[dict]) -> tuple[dict, float, str]:
    """Le petit modèle lit toute la conversation et renvoie la fiche. Retourne (fiche propre, durée, réponse brute)."""
    messages = [{"role": "system", "content": CONSIGNE},
                {"role": "user", "content": message_utilisateur(texte_conversation(historique))}]
    t0 = time.perf_counter()
    brut = ollama_chat(modele_slm, messages, json_mode=True, options={"num_predict": 600})
    return nettoyer(lire_json(brut)), time.perf_counter() - t0, brut


def fusionner(ancienne: dict, nouvelle: dict) -> dict:
    """Une valeur trouvée remplace l'ancienne ; une case vide ne supprime pas ce qui était déjà connu."""
    return {c: nouvelle.get(c) if nouvelle.get(c) is not None else ancienne.get(c) for c in NOMS}


def case_suivante(fiche: dict):
    for champ, quoi in A_DEMANDER:
        if fiche.get(champ) is not None:
            continue
        if champ in ("bonus_malus", "sinistres_responsables_36_mois") and fiche.get("deja_assure") is False:
            continue  # sans antécédent d'assurance, pas de bonus-malus ni de sinistres
        return champ, quoi
    return None


def cases_connues(fiche: dict) -> int:
    return sum(1 for c, _ in A_DEMANDER if fiche.get(c) is not None)


def question_suivante(modele_llm: str, fiche: dict, historique: list[dict], quoi: str):
    """Le gros LLM formule la question suivante (réponse en flux, mot par mot)."""
    connu = {c: v for c, v in fiche.items() if v is not None}
    contexte = (f"Ce qu'on sait déjà du client : {json.dumps(connu, ensure_ascii=False)}\n"
                f"Information à demander maintenant : {quoi}.")
    messages = [{"role": "system", "content": CONSIGNE_CONSEILLER + "\n\n" + contexte}] + historique[-8:]
    return ollama_chat(modele_llm, messages, stream=True, options={"temperature": 0.4})


def recommandation(modele_llm: str, fiche: dict, offres: list[dict]):
    connu = {c: v for c, v in fiche.items() if v is not None}
    messages = [{"role": "system", "content": CONSIGNE_RECOMMANDATION},
                {"role": "user", "content": f"Fiche du client : {json.dumps(connu, ensure_ascii=False)}\n\n"
                                            f"Offres trouvées : {json.dumps(offres[:5], ensure_ascii=False)}"}]
    return ollama_chat(modele_llm, messages, stream=True, options={"temperature": 0.3})
