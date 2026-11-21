"""
Le cerveau du conseiller.

À chaque message du client :
  1. le PETIT modèle (Ollama) lit toute la conversation et propose une fiche JSON (consigne de schema.py) ;
  2. le code garde seulement les valeurs autorisées ET présentes dans ce que le client a dit (pas d'invention) ;
  3. la réponse du client à la dernière question est aussi lue directement par des règles simples, pour la case demandée ;
  4. le code choisit la question suivante (jamais une question déjà répondue, jamais hors liste) ;
  5. le GROS LLM (Ollama) la reformule poliment et répond si le client a posé une question ;
  6. quand la fiche est complète, le code cherche les offres et le gros LLM explique la recommandation.
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
import unicodedata
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from schema import CHAMPS, CONSIGNE, NOMS, message_utilisateur  # noqa: E402

OLLAMA = os.environ.get("OLLAMA_URL", "http://localhost:11434")
ANNEE = 2026

# --------------------------------------------------------------------------- les questions, dans l'ordre
QUESTIONS = [
    ("type_vehicule", "Souhaitez-vous assurer une voiture ou une moto ?"),
    ("marque_modele", "Quels sont la marque et le modèle de votre véhicule ?"),
    ("age", "Quel âge avez-vous ?"),
    ("commune", "Dans quelle ville habitez-vous ?"),
    ("pays_residence", "Dans quel pays se trouve cette ville ?"),
    ("type_permis", "Quel permis avez-vous : B, A, A2… ?"),
    ("annees_permis", "Depuis quelle année avez-vous votre permis ?"),
    ("deja_assure", "Avez-vous déjà été assuré en tant que conducteur principal ?"),
    ("bonus_malus", "Quel est votre bonus-malus ? Il figure sur votre relevé d'information (par exemple 0,85 ou 50 % de bonus)."),
    ("sinistres_responsables_36_mois", "Avez-vous eu des sinistres ces trois dernières années ? Si oui, combien dont vous étiez responsable ?"),
    ("usage", "Comment utilisez-vous le véhicule : usage privé, trajets domicile-travail ou usage professionnel ?"),
    ("kilometrage_annuel", "Combien de kilomètres parcourez-vous environ par an ?"),
    ("stationnement_nuit", "Où le véhicule est-il garé la nuit : garage fermé, parking privé ou dans la rue ?"),
    ("formule_souhaitee", "Quelle couverture souhaitez-vous : au tiers, intermédiaire (tiers + vol, incendie, bris de glace) ou tous risques ?"),
    ("budget_mensuel_max", "Quel budget maximum souhaitez-vous consacrer à votre assurance par mois ?"),
]
QUESTION = dict(QUESTIONS)
# formulation plus guidée quand le client n'a pas répondu à la question
AIDE = {
    "age": "Pouvez-vous m'indiquer votre âge en années, par exemple « 34 ans » ?",
    "commune": "Pouvez-vous m'indiquer le nom de votre ville, par exemple « Lyon » ou « Le Mans » ?",
    "marque_modele": "Pouvez-vous m'indiquer la marque et le modèle, par exemple « Renault Clio » ?",
    "annees_permis": "En quelle année avez-vous obtenu votre permis, par exemple « 2018 » ?",
    "kilometrage_annuel": "Combien de kilomètres faites-vous par an environ, par exemple « 12 000 km » ?",
    "budget_mensuel_max": "Quel montant maximum par mois, par exemple « 60 € » ?",
}
# cases sans lesquelles on peut quand même chercher des offres : on n'insiste pas plus de deux fois
FACULTATIVES = {"bonus_malus", "kilometrage_annuel", "pays_residence", "sinistres_responsables_36_mois"}
VILLES_ARRONDISSEMENTS = {"paris": "Paris", "lyon": "Lyon", "marseille": "Marseille"}
ACCUEIL = ("Bonjour, je suis votre conseiller AssurIA. Je vais vous poser quelques questions pour trouver les offres "
           "d'assurance qui vous correspondent. Pour commencer, souhaitez-vous assurer une voiture ou une moto ?")

CONSIGNE_CONSEILLER = """Tu es un conseiller en assurance auto et moto, chaleureux et professionnel. Tu écris en français.
Tu ne dis jamais bonjour et tu ne te présentes jamais : la conversation est déjà commencée.
On te donne le dernier message du client et la question que tu dois poser maintenant.
Écris 1 ou 2 phrases courtes :
- si le client a posé une question, réponds-y en une phrase (sans citer de prix ni d'assureur) ;
- sinon, remercie-le en quelques mots, ou pas du tout ;
- puis pose la question donnée, presque mot pour mot. Ne pose aucune autre question.
N'écris que le message destiné au client."""

CONSIGNE_RECOMMANDATION = """Tu es un conseiller en assurance auto et moto. Tu écris en français, sans dire bonjour.
On te donne la formule choisie par le client et les offres trouvées dans notre base, déjà classées.
En 4 à 6 phrases : recommande la première offre (assureur, produit, prix estimé par mois, dans le budget ou non), puis cite
brièvement les deux suivantes comme alternatives, avec leur prix.
Règles strictes : n'utilise QUE les informations données ; ne cite une garantie que si elle figure dans la liste « garanties »
de cette offre ; ne parle jamais de franchise ni de remise ; recopie les prix exactement. Termine en rappelant que les prix
sont des estimations et qu'un devis auprès de l'assureur est nécessaire. Pas de liste à puces, pas de titre."""


# --------------------------------------------------------------------------- outils texte
def simple(v) -> str:
    s = unicodedata.normalize("NFKD", str(v)).encode("ascii", "ignore").decode().lower()
    return re.sub(r"\s+", " ", s.replace("’", "'")).strip()


def nombres(t: str) -> list[int]:
    t = re.sub(r"(\d)[\s.](\d{3})\b", r"\1\2", t)  # 12 000 / 12.000 -> 12000
    vals = []
    for n, k in re.findall(r"(\d+)\s*(k\b)?", t):
        vals.append(int(n) * (1000 if k else 1))
    return vals


MOTS_NOMBRES = {"zero": 0, "aucun": 0, "un": 1, "une": 1, "deux": 2, "trois": 3, "quatre": 4, "cinq": 5}
PAYS = ["France", "Allemagne", "Italie", "Espagne", "Royaume-Uni", "Pays-Bas", "Belgique", "Pologne", "Portugal", "Autriche",
        "Suisse", "Suède", "Irlande", "Grèce", "Roumanie", "Tchéquie", "Norvège", "Danemark", "Finlande", "Luxembourg"]
MARQUES = ["renault", "peugeot", "citroen", "dacia", "toyota", "volkswagen", "vw", "ford", "opel", "fiat", "seat", "skoda",
           "audi", "bmw", "mercedes", "tesla", "kia", "hyundai", "nissan", "mazda", "honda", "suzuki", "volvo", "mini",
           "ds", "alfa", "jeep", "mg", "byd", "cupra", "lancia", "smart", "porsche", "land rover", "range", "lexus",
           "mitsubishi", "subaru", "yamaha", "kawasaki", "ducati", "harley", "ktm", "triumph", "vespa", "piaggio", "kymco",
           "aprilia", "royal enfield", "zero", "sym", "mbk"]
MARQUES_MOTO = ["yamaha", "kawasaki", "ducati", "harley", "ktm", "triumph", "vespa", "piaggio", "kymco", "aprilia",
                "royal enfield", "bmw r", "bmw c", "bmw f", "bmw s", "honda cb", "honda forza", "honda pcx", "honda africa",
                "suzuki gsx", "suzuki v-strom", "zero s", "sym", "peugeot django", "mbk", "moto", "scooter"]


# --------------------------------------------------------------------------- lecture directe d'une réponse
def lire_reponse(champ: str, texte: str) -> dict:
    """Lit la réponse du client à une question précise. Retourne les cases trouvées (peut être vide)."""
    s = simple(texte)
    ns = nombres(s)
    if re.search(r"\b(je ne sais pas|je sais pas|aucune idee|sais plus)\b", s):
        return {}
    if champ == "type_vehicule":
        if re.search(r"\b(moto|scooter|deux roues|2 roues)\b", s):
            return {"type_vehicule": "moto"}
        if re.search(r"\b(voiture|auto|automobile|vehicule)\b", s):
            return {"type_vehicule": "auto"}
    if champ == "marque_modele":
        m = re.sub(r"^(bah |ben |alors |euh |oui )*", "", texte.strip(), flags=re.I)
        m = re.sub(r"^(je (veux|voudrais|souhaite) assurer|assurer|j'ai|j ai|c'est|c est|il s'agit d'|il s'agit de|"
                   r"ma voiture c'est|ma moto c'est)\s+", "", m, flags=re.I)
        m = re.sub(r"^(une|un|ma|mon|la|le)\s+", "", m, flags=re.I)
        m = re.sub(r"\b(voiture|moto|scooter)\b\s*", "", m, flags=re.I).strip(" .,!")
        m = re.sub(r"^(une|un|ma|mon|la|le)\s+", "", m, flags=re.I).strip(" .,!")
        m = re.split(r",| de \d{4}| \d{4}", m)[0].strip()
        if 2 <= len(m) <= 40 and "?" not in m:
            return {"marque_modele": " ".join(w.upper() if any(c.isdigit() for c in w) and len(w) <= 4 else w if w.isupper() else w.capitalize() for w in m.split())}
    if champ == "age":
        ages = [n for n in ns if 14 <= n <= 99]
        if ages and not any(n > 1900 for n in ns):
            return {"age": ages[0]}
        # une année de naissance donnée à la place de l'âge (« de 2004 », « né en 2004 ») : on calcule l'âge,
        # sauf si la phrase parle clairement du véhicule
        naissance = [n for n in ns if ANNEE - 99 <= n <= ANNEE - 14]
        if naissance and not re.search(r"voiture|moto|vehicule|circulation|achete", s):
            return {"age": ANNEE - naissance[0]}
    if champ == "commune":
        if "arrondissement" in s or re.search(r"\b\d{1,2} ?(e|eme|er)\b", s):
            ville = next((v for k, v in VILLES_ARRONDISSEMENTS.items() if k in s), None)
            n = [x for x in ns if 1 <= x <= 20]
            # un arrondissement sans ville : on ne devine pas, on redemandera la ville
            return {"commune": f"{ville} {n[0]}e" if n else ville, "pays_residence": "France"} if ville else {}
        c = re.sub(r"^(j'habite|j habite|je vis|je suis|j'suis|on habite)\s*((aux|au|à|a|sur|dans|en)\b)?\s*", "", texte.strip(), flags=re.I)
        c = re.sub(r"^(aux|au|à|a|sur|dans)\s+", "", c, flags=re.I).strip(" .,!")
        c = re.split(r",| en | au | dans ", c)[0].strip()
        trouve = {}
        if 2 <= len(c) <= 40 and "?" not in c and not re.search(r"\d{3,}", c):
            trouve["commune"] = "Le Mans" if simple(c) in ("mans", "le mans") else c[:1].upper() + c[1:]
        trouve.update(lire_reponse("pays_residence", texte))
        return trouve
    if champ == "pays_residence":
        for p in PAYS:
            if simple(p) in s:
                return {"pays_residence": p}
    if champ == "type_permis":
        m = re.search(r"\b(am|a1|a2|a|b)\b", s.replace("permis", " "))
        if m:
            return {"type_permis": m.group(1).upper()}
    if champ == "annees_permis":
        if re.search(r"(viens de l'avoir|cette annee|il y a quelques mois|tout juste)", s):
            return {"annees_permis": 0}
        an = [n for n in ns if 1950 <= n <= ANNEE]
        if an:
            return {"annees_permis": ANNEE - an[0]}
        n = [n for n in ns if 0 <= n <= 70]
        if n:
            return {"annees_permis": n[0]}
    if champ == "deja_assure":
        if re.search(r"\b(non|jamais|pas encore|premiere)\b", s):
            return {"deja_assure": False, "annees_assure_total": 0}
        if re.search(r"\b(oui|ouais|bien sur|depuis)\b", s):
            r = {"deja_assure": True}
            n = [n for n in ns if 0 < n <= 70]
            if n:
                r["annees_assure_total"] = n[0]
            return r
    if champ == "bonus_malus":
        m = re.search(r"\b([0-3][.,]\d{1,2})\b", s)
        if m:
            return {"bonus_malus": f"CRM {float(m.group(1).replace(',', '.')):.2f}"}
        m = re.search(r"(\d{1,2})\s*%?\s*(de )?(bonus|malus)|(bonus|malus) de (\d{1,2})", s)
        if m:
            p = int(m.group(1) or m.group(5))
            sens = m.group(3) or m.group(4)
            return {"bonus_malus": f"CRM {1 - p / 100 if sens == 'bonus' else 1 + p / 100:.2f}"}
        if re.search(r"(classe|degre|sf|stufe|trede|no claims|ans? sans sinistre)", s):
            return {"bonus_malus": texte.strip(" .")}
    if champ == "sinistres_responsables_36_mois":
        if re.search(r"\b(aucun|non|jamais|rien|zero|pas de sinistre)\b", s) and not ns:
            return {"sinistres_responsables_36_mois": 0, "sinistres_non_responsables_36_mois": 0}
        mots = [MOTS_NOMBRES[w] for w in re.findall(r"\b(" + "|".join(MOTS_NOMBRES) + r")\b", s)]
        vals = ns or mots
        if vals and re.search(r"responsable|ma faute|en tort", s) and not re.search(r"(pas|non) (de ma faute|responsable)", s):
            return {"sinistres_responsables_36_mois": vals[0]}
        if vals and re.search(r"(pas|non) (de ma faute|responsable)", s):
            return {"sinistres_responsables_36_mois": 0, "sinistres_non_responsables_36_mois": vals[0]}
        if vals:
            return {"sinistres_responsables_36_mois": vals[0]}
    if champ == "usage":
        for mot, v in [(r"vtc", "vtc"), (r"travail|boulot|bureau|taf", "prive_trajet_travail"),
                       (r"\bpro\b|professionn|clients?|metier|tournee", "professionnel"),
                       (r"loisir|balade|week|vacances", "loisirs"), (r"prive|perso|courses|famille", "prive")]:
            if re.search(mot, s):
                return {"usage": v}
    if champ == "kilometrage_annuel":
        km = [n for n in ns if 300 <= n <= 150000]
        if km:
            return {"kilometrage_annuel": km[0]}
    if champ == "stationnement_nuit":
        for mot, v in [(r"garage|box", "garage_ferme"), (r"parking|residence|cour|place|allee|jardin", "parking_prive"),
                       (r"rue|trottoir|dehors|voie publique|exterieur", "voie_publique")]:
            if re.search(mot, s):
                return {"stationnement_nuit": v}
    if champ == "formule_souhaitee":
        for mot, v in [(r"tous risques|tout risque|meilleure|complet|maximum|bien couvert", "tous_risques"),
                       (r"tiers ?\+|tiers plus|intermediaire|etendu|vol", "tiers_plus"),
                       (r"tiers|minimum|moins cher|basique|responsabilite civile", "tiers")]:
            if re.search(mot, s):
                return {"formule_souhaitee": v}
    if champ == "budget_mensuel_max":
        b = [n for n in ns if 5 <= n <= 2000]
        if b:
            devise = "GBP" if re.search(r"£|livre|pound", s) else "CHF" if re.search(r"chf|franc", s) else "EUR"
            return {"budget_mensuel_max": b[0], "devise_budget": devise}
    return {}


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
                yield json.loads(ligne).get("message", {}).get("content", "")
    return morceaux()


# --------------------------------------------------------------------------- nettoyage de la fiche du petit modèle
VALEURS = {c: [v.strip() for v in d.split("|")] for c, (t, d) in CHAMPS.items() if t == "texte" and "|" in d}
BORNES = {"age": (14, 99), "annees_permis": (0, 70), "annees_assure_total": (0, 70), "kilometrage_annuel": (300, 150000),
          "annee_mise_en_circulation": (1950, ANNEE), "budget_mensuel_max": (5, 2000),
          "sinistres_responsables_36_mois": (0, 10), "sinistres_non_responsables_36_mois": (0, 10)}


# mots qui doivent accompagner un nombre dans le même message du client pour qu'il serve à cette case
INDICES = {"age": r"\bans?\b|age|ne en|\b(19|20)\d\d\b", "kilometrage_annuel": r"km|kilom",
           "budget_mensuel_max": r"€|euro|eur\b|par mois|/mois|budget|£|chf|livre", "annees_permis": r"permis|depuis|obtenu",
           "annee_mise_en_circulation": r"de (19|20)\d\d|circulation|date|annee", "annees_assure_total": r"assure|depuis"}


def nombre_soutenu(champ: str, v: int, messages: list[str]) -> bool:
    for m in messages:
        s = simple(m)
        if (v in nombres(s) or ANNEE - v in nombres(s)) and re.search(INDICES.get(champ, r"."), s):
            return True
    return False


def nettoyer(brut: dict, texte_client: str, messages: list[str] | None = None) -> dict:
    """Garde une valeur seulement si elle est autorisée, plausible, et présente dans ce que le client a écrit."""
    client = simple(texte_client)
    chiffres = set(nombres(client))
    propre = {}
    for c in NOMS:
        v = brut.get(c)
        if v is None or simple(v) in ("", "null", "none", "inconnu", "n/a"):
            continue
        t = CHAMPS[c][0]
        try:
            if t == "entier":
                v = int(float(str(v).replace(" ", "").replace(",", ".")))
                if c == "age" and ANNEE - 99 <= v <= ANNEE - 14:  # le modèle a recopié l'année de naissance
                    v = ANNEE - v
                lo, hi = BORNES.get(c, (0, 10 ** 6))
                # un nombre doit apparaître dans les messages du client (ou s'en déduire pour les années)
                if not lo <= v <= hi or (v not in chiffres and (ANNEE - v) not in chiffres and v > 5):
                    continue
                petit_ok = c.startswith("sinistres") and v <= 5  # « aucun sinistre » donne 0 sans chiffre écrit
                if messages is not None and not petit_ok and not nombre_soutenu(c, v, messages):
                    continue  # le nombre existe, mais le client parlait d'autre chose
            elif t == "booléen":
                v = v if isinstance(v, bool) else simple(v) in ("true", "oui", "vrai", "1")
            elif c in VALEURS:
                ok = {simple(x).replace(" ", "_"): x for x in VALEURS[c]}
                v = ok.get(simple(v).replace(" ", "_"))
                if v is None:
                    continue
                # pour une case que l'on demande, la valeur doit se retrouver dans ce que le client a écrit
                if c in QUESTION and lire_reponse(c, texte_client).get(c) not in (None, v):
                    continue
                if c in QUESTION and lire_reponse(c, texte_client).get(c) is None and c != "deja_assure":
                    continue
            else:
                v = str(v).strip()
                if c in ("commune", "marque_modele") and simple(v).split(" ")[0] not in client:
                    continue
                if c == "commune" and (re.search(r"arrondissement|^\d", simple(v)) or len(v) > 40):
                    continue
        except (ValueError, TypeError):
            continue
        propre[c] = v
    return propre


def lire_json(texte: str) -> dict:
    debut, fin = texte.find("{"), texte.rfind("}")
    try:
        d = json.loads(texte[debut:fin + 1])
        return d if isinstance(d, dict) else {}
    except (json.JSONDecodeError, ValueError):
        return {}


# --------------------------------------------------------------------------- la session de conversation
class Conversation:
    def __init__(self, modele_slm: str, modele_llm: str):
        self.slm, self.llm = modele_slm, modele_llm
        self.historique = [{"role": "assistant", "content": ACCUEIL}]
        self.fiche = {c: None for c in NOMS}
        self.demandee = "type_vehicule"
        self.essais: dict[str, int] = {}
        self.ignorees: set[str] = set()
        self.terminee = False
        self.journal: list[dict] = []

    def texte(self) -> str:
        return "\n".join(f"{'Conseiller' if m['role'] == 'assistant' else 'Client'} : {m['content']}" for m in self.historique)

    def texte_client(self) -> str:
        return " ".join(m["content"] for m in self.historique if m["role"] == "user")

    def completer(self, deduire):
        """Déductions sûres : pays et type de véhicule depuis la base, pas de bonus sans antécédent."""
        f = self.fiche
        if f["commune"] and not f["pays_residence"]:
            f["pays_residence"] = deduire("pays", f["commune"])
        if f["marque_modele"] and not f["type_vehicule"]:
            t = deduire("type", f["marque_modele"])
            if t is None and any(m in simple(f["marque_modele"]) for m in MARQUES_MOTO):
                t = "moto"
            f["type_vehicule"] = t

    def case_suivante(self):
        for champ, _ in QUESTIONS:
            if self.fiche.get(champ) is not None or champ in self.ignorees:
                continue
            if champ in ("bonus_malus", "sinistres_responsables_36_mois") and self.fiche.get("deja_assure") is False:
                continue
            return champ
        return None

    def progression(self) -> tuple[int, int]:
        utiles = [c for c, _ in QUESTIONS if not (c in ("bonus_malus", "sinistres_responsables_36_mois")
                                                    and self.fiche.get("deja_assure") is False)]
        return sum(1 for c in utiles if self.fiche.get(c) is not None or c in self.ignorees), len(utiles)

    PAIRES = {"type_vehicule": ["marque_modele"], "marque_modele": ["type_vehicule"], "type_permis": ["annees_permis"],
              "annees_permis": ["type_permis"], "deja_assure": [], "commune": ["pays_residence"]}

    def tour(self, message: str, deduire) -> dict:
        """Traite un message : met la fiche à jour et choisit la suite. Retourne ce qu'il faut afficher."""
        self.historique.append({"role": "user", "content": message})
        # 1. le petit modèle propose une fiche à partir de toute la conversation
        t0 = time.perf_counter()
        brut, erreur = "", None
        try:
            brut = ollama_chat(self.slm, [{"role": "system", "content": CONSIGNE},
                                          {"role": "user", "content": message_utilisateur(self.texte())}],
                               json_mode=True, options={"num_predict": 600})
        except Exception as e:  # le conseiller continue avec les règles seules
            erreur = str(e)
        duree = time.perf_counter() - t0
        propose = nettoyer(lire_json(brut), self.texte_client(),
                           [m["content"] for m in self.historique if m["role"] == "user"])
        for c, v in propose.items():
            self.fiche[c] = v
        # 2. la réponse à la question posée, lue directement
        direct = lire_reponse(self.demandee, message) if self.demandee else {}
        for voisin in self.PAIRES.get(self.demandee, []):  # une réponse donne souvent aussi la case voisine
            if self.fiche.get(voisin) is None:
                if voisin == "marque_modele" and not any(re.search(rf"\b{m}\b", simple(message)) for m in MARQUES):
                    continue  # on ne prend un modèle « en passant » que si une marque est citée
                for c, v in lire_reponse(voisin, message).items():
                    direct.setdefault(c, v)
        for c, v in direct.items():
            self.fiche[c] = v
        self.completer(deduire)
        if self.fiche.get("deja_assure") is False:
            self.fiche["annees_assure_total"] = 0
        # 3. la question a-t-elle reçu une réponse ?
        compris = self.demandee is None or self.fiche.get(self.demandee) is not None
        if not compris:
            self.essais[self.demandee] = self.essais.get(self.demandee, 0) + 1
            if self.essais[self.demandee] >= 2 and self.demandee in FACULTATIVES:
                self.ignorees.add(self.demandee)  # case facultative : on n'insiste pas plus de deux fois
        self.journal.append({"petit_modele": self.slm, "duree_s": round(duree, 2), "reponse_brute": brut,
                             "retenu_du_petit_modele": propose, "lu_dans_la_reponse": direct, "erreur": erreur})
        suivante = self.case_suivante()
        relance = not compris and suivante == self.demandee
        self.demandee = suivante
        return {"suivante": suivante, "relance": relance}

    def question(self, suivante: str, relance: bool, message: str) -> str:
        """Le gros LLM formule la question ; si sa réponse sort du cadre, on garde la question prévue."""
        prevue = QUESTION[suivante]
        if relance and self.essais.get(suivante, 0) >= 1 and suivante in AIDE:
            prevue = AIDE[suivante]
        elif suivante == "marque_modele" and self.fiche.get("type_vehicule"):
            prevue = f"Quels sont la marque et le modèle de votre {'moto' if self.fiche['type_vehicule'] == 'moto' else 'voiture'} ?"
        # « je n'ai pas compris » seulement si le client a vraiment tenté de répondre (pas pour un simple bonjour)
        salutation = re.search(r"\b(bonjour|salut|bonsoir|hello|devis|offre|assurance|aide)\b", simple(message))
        if relance and suivante == "commune" and "arrondissement" in simple(message):
            prevue = "Dans quelle ville se trouve cet arrondissement : Paris, Lyon, Marseille ?"
        elif relance and suivante == "kilometrage_annuel" and any(n > 150000 for n in nombres(simple(message))):
            prevue = "Ce kilométrage me paraît très élevé pour une année. " + prevue
        elif relance and not salutation:
            prevue = "Je n'ai pas bien saisi votre réponse. " + prevue
        if os.environ.get("QUESTIONS_LLM", "1") == "0":
            return prevue
        try:
            texte = ollama_chat(self.llm, [
                {"role": "system", "content": CONSIGNE_CONSEILLER},
                {"role": "user", "content": f"Dernier message du client : « {message} »\nQuestion à poser : « {prevue} »"}],
                options={"temperature": 0.3, "num_predict": 120}).strip()
        except Exception:
            return prevue
        texte = re.sub(r"<think>.*?</think>", "", texte, flags=re.S).strip().strip('"«» ')
        mots_cles = [w for w in re.findall(r"\w{5,}", simple(prevue)) if w not in ("souhaitez", "quelle", "quels")]
        cadre = (texte.endswith("?") and len(texte) < 350 and texte.count("?") == 1
                 and not re.search(r"\b(bonjour|je suis votre)\b", simple(texte))
                 and any(m in simple(texte) for m in mots_cles[:4]))
        return texte if cadre else prevue

    def recommandation(self, offres: list[dict]):
        resume = [{"rang": i + 1, "assureur": o["assureur"], "produit": o["produit"],
                   "prix_mensuel_estime": f"{o['prix_mensuel_estime_eur']:.2f} €".replace(".", ","),
                   "dans_le_budget": o["dans_le_budget"], "garanties": o["garanties"]} for i, o in enumerate(offres[:3])]
        client = {"formule": self.fiche.get("formule_souhaitee"), "budget_par_mois": self.fiche.get("budget_mensuel_max")}
        return ollama_chat(self.llm, [
            {"role": "system", "content": CONSIGNE_RECOMMANDATION},
            {"role": "user", "content": f"Client : {json.dumps(client, ensure_ascii=False)}\n\n"
                                        f"Offres (déjà classées) : {json.dumps(resume, ensure_ascii=False)}"}],
            stream=True, options={"temperature": 0})
