"""
Conversations d'entraînement calquées sur l'application (version 2 des données).

Le premier générateur (generer_conversations.py) posait ses propres questions, souvent plusieurs à la fois, et le client
répondait par des phrases complètes. L'application, elle, pose une question à la fois, dans son ordre, et les vrais
clients répondent court (« 2020 », « Comptant », « le mans »), refusent parfois, font des fautes ou répondent à côté.
Ici, on rejoue exactement le déroulé de l'application (mêmes questions, même ordre, mêmes questions sautées) avec des
clients simulés de styles différents.

Règle d'étiquetage : le profil attendu ne contient que ce que le client a dit (ou ce qui s'en déduit sans doute
possible, comme le pays d'une ville). Une question refusée reste vide.

Comme dans l'application, le petit modèle est appelé après chaque message du client : une partie des exemples est donc
coupée juste après un message du client, avec le profil attendu à ce moment-là.

    python simuler_clients.py                 # écrit donnees/ (train, valid, test, test_conversations)
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import unicodedata
from pathlib import Path

import pandas as pd

RACINE = Path(__file__).parent
sys.path.insert(0, str(RACINE))
sys.path.insert(0, str(RACINE / "app"))
from schema import CONSIGNE, NOMS, message_utilisateur  # noqa: E402
import generer_conversations as g  # noqa: E402  (banques d'expressions et fonctions utiles)
import agent  # noqa: E402  (les questions de l'application, dans son ordre)
from offres import UE  # noqa: E402

PROFILS = RACINE.parent / "assuria-projet" / "data" / "objectif1" / "profils" / "profils.parquet"
SORTIE = RACINE / "donnees"
GRAINE = 2027
ANNEE = 2026
N_TRAIN, N_VALID, N_TEST = 6000, 500, 500

R = random.Random(GRAINE)
g.R = R  # les fonctions du premier générateur tirent aussi dans ce générateur
un = R.choice


def chance(p: float) -> bool:
    return R.random() < p


# --------------------------------------------------------------------------- styles de clients
# court : « 2020 », « comptant » ; phrase : réponses complètes ; familier : sans majuscules, fautes ;
# bavard : ajoute des informations ; hesitant : refuse ou ne sait pas plus souvent
STYLES = [("court", 0.30), ("phrase", 0.30), ("familier", 0.20), ("bavard", 0.12), ("hesitant", 0.08)]
REFUS = ["Je préfère ne pas répondre.", "je préfère pas dire", "Je ne souhaite pas le préciser.", "c'est perso ça",
         "Je préfère ne pas le dire.", "pas envie de répondre à ça", "je veux pas le dire", "Je passe.",
         "je veux pas mentionner", "je préfère pas répondre à cette question", "Joker."]
NE_SAIT_PAS = ["Je ne sais pas.", "aucune idée", "je sais pas trop", "Je ne sais plus, il faudrait que je vérifie.",
               "euh je sais pas", "Je n'ai pas l'info sous la main."]
POLI = ["", "", "", "Merci. ", "Merci beaucoup. ", "Très bien. ", "D'accord. ", "Parfait. ", "Entendu. ", "C'est noté. "]
FAUTES = {"voiture": "voitur", "depuis": "depui", "garage": "garrage", "comptant": "contant", "jamais": "jamai",
          "assurance": "assurence", "parking": "parkin", "travail": "travaille", "crédit": "credi", "franchise": "franchize",
          "tous risques": "tout risque", "ans": "an", "kilomètres": "kilometre", "environ": "envirron",
          "personne": "persone", "privé": "privee", "obtenu": "obtenue", "essence": "essance", "diesel": "diésel",
          "électrique": "electrik", "hybride": "hybrid", "moins": "moin", "maximum": "maximun", "standard": "standart"}


def sans_accents(t: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", t) if unicodedata.category(c) != "Mn")


def style_ecrit(t: str, style: str) -> str:
    """Met la réponse dans le style du client (majuscule, point, minuscules, fautes)."""
    t = t.strip()
    if style == "familier":
        t = t.lower()
        for juste, faux in FAUTES.items():
            if juste in t and chance(0.35):
                t = t.replace(juste, faux, 1)
        if chance(0.5):
            t = sans_accents(t)
        return t.rstrip(".")
    if style == "court" and chance(0.5):
        return t[:1].upper() + t[1:] if chance(0.6) else t
    t = t[:1].upper() + t[1:]
    return t if t.endswith((".", "!", "?")) or chance(0.3) else t + "."


# --------------------------------------------------------------------------- réponses, question par question
# chaque fonction : (profil, style) -> (texte, {case: valeur}) ; None = le client refuse / ne sait pas
def r_type(p, s):
    v = p.type_vehicule
    nom = "une moto" if v == "moto" else "une voiture"
    if "scooter" in str(p.categorie_vehicule) and chance(0.5):
        nom = "un scooter"
    t = {"court": [nom, "moto" if v == "moto" else "voiture", nom.split()[-1]],
         "phrase": [f"Je voudrais assurer {nom}.", f"C'est pour {nom}.", f"Bonjour, {nom}."],
         "familier": [f"{nom} stp", f"c pour {nom}", nom.split()[-1]],
         "bavard": [f"Bonjour, c'est pour {nom} que je viens d'acheter.", f"{nom}, je change d'assureur."],
         "hesitant": [nom, f"euh {nom}"]}[s]
    return un(t), {"type_vehicule": v}


def r_marque(p, s):
    mm = p.marque_modele
    a = int(p.annee_mise_en_circulation)
    champs = {"marque_modele": mm}
    t = un({"court": [mm, mm.lower(), f"une {mm}"],
            "phrase": [f"C'est une {mm}.", f"Une {mm}.", f"J'ai une {mm}."],
            "familier": [mm.lower(), f"une {mm.lower()}", f"{mm.lower()}"],
            "bavard": [f"Une {mm}, je l'adore.", f"C'est une {mm}, achetée il y a peu."],
            "hesitant": [f"une {mm} je crois", mm]}[s])
    if s in ("phrase", "bavard") and chance(0.3):
        t = t.rstrip(".") + f" de {a}."
        champs["annee_mise_en_circulation"] = a
    return t, champs


def r_annee(p, s):
    a = int(p.annee_mise_en_circulation)
    age_v = ANNEE - a
    options = {"court": [f"{a}", f"de {a}", f"{a}"],
               "phrase": [f"Elle date de {a}.", f"Première mise en circulation en {a}.", f"De {a}."],
               "familier": [f"{a}", f"de {a}", f"elle est de {a}"],
               "bavard": [f"Elle est de {a}, je l'ai eue d'occasion.", f"{a}, c'est écrit sur la carte grise."],
               "hesitant": [f"{a} je crois", f"je pense {a}"]}[s]
    if age_v == 0:
        options += ["elle est neuve", "neuve, de cette année"]
    elif 1 <= age_v <= 15 and s in ("phrase", "bavard"):
        options.append(f"Elle a {g.ans(age_v)}.")
    return un(options), {"annee_mise_en_circulation": a}


ENERGIE_COURT = {"essence": ["essence", "sans plomb", "SP95"], "diesel": ["diesel", "gasoil", "gazole"],
                 "hybride": ["hybride"], "electrique": ["électrique", "100 % électrique"]}


def r_energie(p, s):
    e = p.energie
    t = un(ENERGIE_COURT[e]) if s in ("court", "familier", "hesitant") else un(g.ENERGIE[e])
    return t, {"energie": e}


ACQ_COURT = {"comptant": ["comptant", "cash", "payée comptant"], "credit": ["crédit", "à crédit", "avec un prêt"],
             "LOA_LLD": ["LOA", "en LOA", "LLD", "leasing"]}


def r_acquisition(p, s):
    m = p.mode_acquisition
    t = un(ACQ_COURT[m]) if s in ("court", "familier", "hesitant") else un(g.ACQUISITION[m])
    return t, {"mode_acquisition": m}


def r_age(p, s):
    a = int(p.age)
    options = {"court": [f"{a}", f"{a} ans"], "phrase": [f"J'ai {a} ans.", f"{a} ans."],
               "familier": [f"{a}", f"{a} ans", f"j'ai {a}"], "bavard": [f"J'ai {a} ans, bientôt {a + 1}.", f"{a} ans."],
               "hesitant": [f"{a}", f"{a} ans"]}[s]
    if chance(0.12):  # année de naissance au lieu de l'âge
        options = [f"je suis né en {ANNEE - a}", f"{ANNEE - a}", f"je suis de {ANNEE - a}"]
    return un(options), {"age": a}


def r_commune(p, s):
    c = p.commune
    options = {"court": [c, c.lower(), f"à {c}"], "phrase": [f"J'habite à {c}.", f"À {c}.", f"Je vis à {c}."],
               "familier": [c.lower(), f"a {c.lower()}", f"j'habite {c.lower()}"],
               "bavard": [f"À {c}, {g.prep_pays(p.pays_residence)}.", f"J'habite à {c} depuis quelques années."],
               "hesitant": [c, f"à {c}"]}[s]
    t = un(options)
    champs = {"commune": c, "pays_residence": p.pays_residence}  # le pays se déduit de la ville
    return t, champs


def r_pays(p, s):
    pays = p.pays_residence
    return un([pays, pays.lower(), f"{g.prep_pays(pays)}", f"J'habite {g.prep_pays(pays)}."]), {"pays_residence": pays}


def r_type_permis(p, s):
    tp = g.permis_normalise(p.type_permis)
    nom = {"B+formation": "B avec la formation 125"}.get(tp, tp)
    t = un({"court": [nom, f"permis {nom}", f"le {nom}"], "phrase": [f"J'ai le permis {nom}.", f"Le permis {nom}."],
            "familier": [f"permis {nom.lower()}", nom.lower(), f"le {nom.lower()}"],
            "bavard": [f"Le permis {nom}, je l'ai eu du premier coup.", f"Permis {nom}."], "hesitant": [nom, f"le {nom}"]}[s])
    return t, {"type_permis": tp}


def r_annees_permis(p, s):
    n = int(p.annees_permis)
    an = ANNEE - n
    options = {"court": [f"{an}", f"{an}", f"depuis {an}", f"de {an}"],
               "phrase": [f"Depuis {an}.", f"Je l'ai obtenu en {an}.", f"Ça fait {g.ans(n)}."],
               "familier": [f"{an}", f"depuis {an}", f"ca fait {g.ans(n)}", f"en {an}"],
               "bavard": [f"Depuis {an}, ça fait {g.ans(n)}.", f"Je l'ai eu en {an}."], "hesitant": [f"{an} je crois", f"vers {an}"]}[s]
    if n == 0:
        options = ["cette année", "je viens de l'avoir", f"{ANNEE}", "il y a quelques mois"]
    elif n == 1:
        options += ["l'année dernière", f"{ANNEE - 1}"]
    age_obtention = int(p.age) - n
    if 16 <= age_obtention <= 30 and chance(0.08):
        options = [f"je l'ai eu à {age_obtention} ans"]  # avec l'âge déjà donné, la durée se calcule
    return un(options), {"annees_permis": n}


def r_pays_permis(p, s):
    pp, pr = p.pays_permis, p.pays_residence
    if pp == pr:
        options = [f"{g.prep_pays(pp)}", "ici", "dans mon pays", "là où j'habite", pp]
        return un(options), {"pays_permis": pp}
    if chance(0.12):  # sans nommer le pays
        return (un(["dans un autre pays de l'UE", "ailleurs en Europe"]), {"pays_permis": "Union européenne"}) if pp in UE else \
            (un(["hors de l'Union européenne", "hors UE"]), {"pays_permis": "Hors Union européenne"})
    return un([f"{g.prep_pays(pp)}", f"non, {g.prep_pays(pp)}", f"je l'ai passé {g.prep_pays(pp)}", pp]), {"pays_permis": pp}


def r_deja_assure(p, s):
    if not p.deja_assure:
        t = un({"court": ["non", "non jamais", "jamais"], "phrase": ["Non, jamais.", "Non, ce sera ma première assurance."],
                "familier": ["non jamais", "nn", "non c'est la premiere fois"],
                "bavard": ["Non, j'étais sur l'assurance de mes parents.", "Non, jamais à mon nom."], "hesitant": ["non", "je crois pas"]}[s])
        return t, {"deja_assure": False, "annees_assure_total": 0}
    n = int(p.annees_assure_total)
    if s == "court" or chance(0.35):
        return un(["oui", "Oui.", "ouais"]), {"deja_assure": True}
    return un([f"Oui, depuis {g.ans(n)}.", f"oui ca fait {g.ans(n)}", f"Oui, {g.ans(n)} d'assurance."]), \
        {"deja_assure": True, "annees_assure_total": n}


def r_conduite_accompagnee(p, s):  # posée par l'application, mais pas dans la fiche du petit modèle
    oui = bool(p.conduite_accompagnee)
    return un(["oui", "Oui, en conduite accompagnée.", "oui AAC"] if oui else ["non", "Non.", "non, permis classique"]), {}


def r_bonus(p, s):
    return g.bonus_dit(p), {"bonus_malus": p.niveau_bonus_malus_national}


def r_sinistres(p, s):
    r, nr = int(p.sinistres_responsables_36_mois), int(p.sinistres_non_responsables_36_mois)
    if r == 0 and nr == 0 and s in ("court", "familier"):
        return un(["non", "aucun", "0", "non aucun", "rien"]), {"sinistres_responsables_36_mois": 0, "sinistres_non_responsables_36_mois": 0}
    return g.sinistres_dit(r, nr), {"sinistres_responsables_36_mois": r, "sinistres_non_responsables_36_mois": nr}


def r_resiliation(p, s):
    v = p.resiliation_assureur
    return un(g.RESILIATION[v] + (["non", "jamais"] if v == "non" else [])), {"resiliation_assureur": v}


SECOND_COURT = {"aucun": ["oui", "oui je suis seul", "que moi", "oui, seul", "juste moi"],
                "conjoint": ["non, ma femme aussi", "mon mari aussi", "non ma copine aussi", "ma conjointe aussi"],
                "enfant_jeune_conducteur": ["non, mon fils aussi, il vient d'avoir le permis", "ma fille, jeune conductrice"],
                "parent": ["non, mon père aussi", "ma mère de temps en temps"]}


def r_secondaire(p, s):
    v = p.conducteur_secondaire
    t = un(SECOND_COURT[v]) if s in ("court", "familier", "hesitant") else un(g.SECONDAIRE[v] + SECOND_COURT[v])
    return t, {"conducteur_secondaire": v}


USAGE_COURT = {"prive": ["privé", "perso", "usage privé"], "prive_trajet_travail": ["trajet travail", "pour aller au boulot", "domicile travail"],
               "professionnel": ["pro", "professionnel", "pour le travail", "un peu de tout", "les trois"],
               "loisirs": ["loisirs", "juste le week-end"], "vtc": ["VTC"]}


def r_usage(p, s):
    v = p.usage
    t = un(USAGE_COURT[v]) if s in ("court", "familier", "hesitant") else un(g.USAGE[v] + USAGE_COURT[v][:2])
    return t, {"usage": v}


def r_km(p, s):
    k = int(p.kilometrage_annuel)
    t = un({"court": [f"{k}", g.nombre(k), f"{g.nombre(k)} km"], "phrase": [f"Environ {g.nombre(k)} km par an.", f"À peu près {g.nombre(k)} km."],
            "familier": [f"{k}", f"{k // 1000}k" if k % 1000 == 0 else f"{k}", f"dans les {g.nombre(k)}"],
            "bavard": [f"Environ {g.nombre(k)} km, surtout pour le travail.", f"Je dirais {g.nombre(k)} km par an."],
            "hesitant": [f"je dirais {g.nombre(k)}", f"peut-être {g.nombre(k)}"]}[s])
    return t, {"kilometrage_annuel": k}


STAT_COURT = {"garage_ferme": ["garage", "au garage", "dans mon garage"], "parking_prive": ["parking", "parking privé", "sur mon parking"],
              "voie_publique": ["dans la rue", "rue", "dehors"]}


def r_stationnement(p, s):
    v = p.stationnement_nuit
    t = un(STAT_COURT[v]) if s in ("court", "familier", "hesitant") else un(g.STATIONNEMENT[v])
    return t, {"stationnement_nuit": v}


FORM_COURT = {"tiers": ["au tiers", "tiers", "le moins cher"], "tiers_plus": ["intermédiaire", "tiers plus", "tiers +"],
              "tous_risques": ["tous risques", "tout risque", "le max"]}


def r_formule(p, s):
    v = p.formule_souhaitee
    t = un(FORM_COURT[v]) if s in ("court", "familier", "hesitant") else un(g.FORMULE[v])
    return t, {"formule_souhaitee": v}


FRAN_COURT = {"basse": ["basse", "la plus basse"], "standard": ["standard", "normale", "peu importe"], "haute": ["haute", "élevée"]}


def r_franchise(p, s):
    v = p.franchise_souhaitee
    t = un(FRAN_COURT[v]) if s in ("court", "familier", "hesitant") else un(g.FRANCHISE[v])
    return t, {"franchise_souhaitee": v}


def r_budget(p, s):
    b, d = int(p.budget_mensuel_max_devise_locale), p.devise_locale
    if s in ("court", "familier") and d == "EUR":
        t = un([f"{b}", f"{b} euros", f"{b}€", f"{b} euro", f"max {b}"])
    else:
        t, _ = g.e_budget(p)
    return t, {"budget_mensuel_max": b, "devise_budget": d}


REPONSES = {"type_vehicule": r_type, "marque_modele": r_marque, "annee_mise_en_circulation": r_annee, "energie": r_energie,
            "mode_acquisition": r_acquisition, "age": r_age, "commune": r_commune, "pays_residence": r_pays,
            "type_permis": r_type_permis, "annees_permis": r_annees_permis, "pays_permis": r_pays_permis,
            "deja_assure": r_deja_assure, "conduite_accompagnee": r_conduite_accompagnee, "bonus_malus": r_bonus,
            "sinistres_responsables_36_mois": r_sinistres, "resiliation_assureur": r_resiliation,
            "conducteur_secondaire": r_secondaire, "usage": r_usage, "kilometrage_annuel": r_km,
            "stationnement_nuit": r_stationnement, "formule_souhaitee": r_formule, "franchise_souhaitee": r_franchise,
            "budget_mensuel_max": r_budget}
NE_SAIT_PAS_POSSIBLE = {"annee_mise_en_circulation", "kilometrage_annuel", "bonus_malus", "energie"}

# ce que le client bavard ajoute spontanément
EXTRAS = [("profession", lambda p: (un(g.PROFESSION[p.profession]), {"profession": p.profession})),
          ("situation_familiale", lambda p: (un(g.SITUATION[p.situation_familiale]), {"situation_familiale": p.situation_familiale})),
          ("paiement", lambda p: ("je préfère payer " + un(g.PAIEMENT[p.paiement]), {"paiement": p.paiement}))]


# --------------------------------------------------------------------------- une conversation
def concerne(champ: str, connu: dict, p) -> bool:
    """Mêmes conditions que l'application (agent.Conversation.concerne)."""
    if champ in ("bonus_malus", "sinistres_responsables_36_mois", "resiliation_assureur"):
        return connu.get("deja_assure") is not False
    if champ == "conduite_accompagnee":
        novice = (connu.get("annees_permis") is not None and connu["annees_permis"] <= 2) or connu.get("deja_assure") is False
        return connu.get("type_vehicule") != "moto" and novice
    if champ == "franchise_souhaitee":
        return connu.get("formule_souhaitee") != "tiers"
    if champ == "pays_residence":  # l'application ne le demande que si la ville n'a pas suffi
        return connu.get("commune") is None or (p.pays_residence != "France" and chance(0.3))
    return True


def conversation(p):
    """Retourne (tours, profils attendus après chaque message du client)."""
    style = R.choices([s for s, _ in STYLES], [w for _, w in STYLES])[0]
    p_refus = {"court": 0.03, "phrase": 0.02, "familier": 0.05, "bavard": 0.02, "hesitant": 0.18}[style]
    profil = {k: None for k in NOMS}
    connu: dict = {}  # ce que l'application sait (pour choisir la question suivante)
    passees: set = set()
    tours = [("Conseiller", agent.ACCUEIL)]
    etats = []

    def client(texte, champs):
        tours.append(("Client", texte))
        profil.update({k: v for k, v in champs.items() if k in profil})
        connu.update(champs)
        etats.append((len(tours), dict(profil)))

    # premier message : parfois plusieurs informations d'un coup
    if chance(0.3):
        morceaux, champs = [], {}
        for c in ["type_vehicule", "marque_modele", "age", "commune", "formule_souhaitee"]:
            if c in ("type_vehicule",) or chance(0.45):
                t, ch = REPONSES[c](p, "court")
                morceaux.append(t if c != "age" else f"j'ai {int(p.age)} ans")
                champs.update(ch)
        debut = un(["Bonjour, ", "Bonjour ", "Salut, ", ""]) + un(["je veux assurer ", "c'est pour ", ""])
        client(style_ecrit(debut + ", ".join(morceaux), style), champs)
    else:
        t, ch = r_type(p, style)
        client(style_ecrit(t, style), ch)
    # la plaque d'immatriculation (voitures) : l'application ajoute ce message pour le client
    plaque = p.type_vehicule == "auto" and chance(0.25)

    relances = {}
    apres_refus = False
    while True:
        suivante = next((c for c, _ in agent.QUESTIONS
                         if connu.get(c) is None and c not in passees and concerne(c, connu, p)), None)
        if suivante is None:
            break
        if suivante == "marque_modele" and plaque:
            tours.append(("Conseiller", agent.QUESTION["marque_modele"]))
            client(f"Mon véhicule est une {p.marque_modele}.", {"marque_modele": p.marque_modele})
            connu.update({"annee_mise_en_circulation": int(p.annee_mise_en_circulation), "energie": p.energie})
            continue
        question = agent.QUESTION[suivante]
        if suivante == "pays_residence" and connu.get("commune") is None:
            question = "Dans quel pays habitez-vous ?"
        if relances.get(suivante):
            question = agent.AIDE.get(suivante, "Je n'ai pas bien saisi votre réponse. " + question)
        debut = "Pas de problème, nous continuons sans cette information. " if apres_refus else (un(POLI) if len(tours) > 2 else "")
        apres_refus = False
        tours.append(("Conseiller", debut + question))
        # refus ou « je ne sais pas »
        if suivante != "type_vehicule" and chance(p_refus):
            refus = un(NE_SAIT_PAS) if suivante in NE_SAIT_PAS_POSSIBLE and chance(0.5) else un(REFUS)
            client(style_ecrit(refus, style), {})
            passees.add(suivante)
            apres_refus = True  # comme l'application : la question suivante commence par « Pas de problème… »
            continue
        # réponse à côté, puis bonne réponse après relance
        if not relances.get(suivante) and chance(0.04):
            q = un(["c'est obligatoire ?", "pourquoi vous demandez ça ?", "ça change le prix ?", "hein ?"])
            client(style_ecrit(q, style), {})
            relances[suivante] = True
            if q != "hein ?":
                tours.append(("Conseiller", un(["C'est un critère que tous les assureurs demandent, il joue sur le prix.",
                                                "Cela permet de calculer un prix juste pour vous."])))
                tours[-1] = ("Conseiller", tours[-1][1] + " " + (agent.AIDE.get(suivante) or agent.QUESTION[suivante]))
                texte, champs = REPONSES[suivante](p, style)
                client(style_ecrit(texte, style), champs)
                if suivante == "conduite_accompagnee":
                    connu["conduite_accompagnee"] = True
            continue
        texte, champs = REPONSES[suivante](p, style)
        if style == "bavard" and chance(0.25):
            nom, f = un(EXTRAS)
            if profil.get(nom) is None:
                extra, ch = f(p)
                texte = texte.rstrip(".") + ", et " + extra
                champs = {**champs, **ch}
        client(style_ecrit(texte, style), champs)
        if suivante == "conduite_accompagnee":
            connu["conduite_accompagnee"] = True  # posée, réponse hors fiche
    return tours, etats, style


def exemple(tours_jusque, profil):
    conv = "\n".join(f"{qui} : {t}" for qui, t in tours_jusque)
    return {"messages": [{"role": "system", "content": CONSIGNE},
                         {"role": "user", "content": message_utilisateur(conv)},
                         {"role": "assistant", "content": json.dumps(profil, ensure_ascii=False)}]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profils", default=str(PROFILS))
    a = ap.parse_args()
    df = pd.read_parquet(a.profils)
    df = df[df.age >= 16]
    tirage = df.sample(N_TRAIN + N_VALID + N_TEST, random_state=GRAINE)
    jeux = {"train": tirage.iloc[:N_TRAIN], "valid": tirage.iloc[N_TRAIN:N_TRAIN + N_VALID], "test": tirage.iloc[N_TRAIN + N_VALID:]}
    SORTIE.mkdir(exist_ok=True)
    styles = {}
    for nom, part in jeux.items():
        lignes, convs = [], []
        for p in part.itertuples():
            tours, etats, style = conversation(p)
            styles[style] = styles.get(style, 0) + 1
            # comme dans l'application : le petit modèle voit la conversation après un message du client
            coupee = chance(0.5) and len(etats) > 2
            n, profil = R.choice(etats[:-1]) if coupee else etats[-1]
            lignes.append(exemple(tours[:n], profil))
            convs.append({"id_profil": p.id_profil, "style": style, "coupee": coupee, "tours": tours[:n], "profil_attendu": profil})
        with open(SORTIE / f"{nom}.jsonl", "w", encoding="utf-8") as f:
            for e in lignes:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")
        if nom == "test":
            with open(SORTIE / "test_conversations.jsonl", "w", encoding="utf-8") as f:
                for c in convs:
                    f.write(json.dumps(c, ensure_ascii=False) + "\n")
        print(f"{nom} : {len(lignes)} conversations")
    print("styles :", styles)
    return 0


if __name__ == "__main__":
    sys.exit(main())
