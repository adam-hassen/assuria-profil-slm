"""Client simulé commun aux deux approches, et notation sur les champs que les deux fiches ont en commun.

Le client a un vrai profil (France, voiture, tiré de profils.parquet). À chaque question, il répond par une phrase
simple et sans ambiguïté, la même quelle que soit l'approche. On note ensuite, pour chaque champ commun :
  - « dit »     : le client a donné l'information pendant la conversation ;
  - « juste »   : la fiche finale contient la bonne valeur ;
  - « inventé » : la fiche contient une valeur alors que le client ne l'a jamais donnée.
"""
from __future__ import annotations

import re
import unicodedata

UE = {"Allemagne", "Autriche", "Belgique", "Bulgarie", "Chypre", "Croatie", "Danemark", "Espagne", "Estonie", "Finlande",
      "Grèce", "Hongrie", "Irlande", "Italie", "Lettonie", "Lituanie", "Luxembourg", "Malte", "Pays-Bas", "Pologne",
      "Portugal", "Roumanie", "Slovaquie", "Slovénie", "Suède", "Tchéquie", "République tchèque"}


def simple(t) -> str:
    return unicodedata.normalize("NFKD", str(t)).encode("ascii", "ignore").decode().lower()


def nombre(x) -> str:
    return f"{int(x):,}".replace(",", " ") if float(x).is_integer() else str(x).replace(".", ",")


# --------------------------------------------------------------------------- réponses du client
def repondre(concept: str, p) -> str:
    z = {"rural": "C'est une commune rurale.", "periurbain": "C'est en périphérie d'une ville, périurbain.",
         "urbain": "C'est une ville moyenne.", "grande_ville": "C'est une grande ville.", "metropole": "C'est une métropole."}
    usage = {"prive": "Usage privé uniquement, pour les loisirs.",
             "prive_trajet_travail": "Pour aller au travail tous les jours, et en privé le week-end.",
             "professionnel": "Usage professionnel, je vois des clients.", "vtc": "Usage professionnel, je fais du VTC."}
    park = {"garage_ferme": "Dans un garage fermé.", "parking_prive": "Sur un parking privé.", "voie_publique": "Dans la rue."}
    formule = {"tiers": "Au tiers.", "tiers_plus": "Une formule intermédiaire, tiers plus vol et bris de glace.",
               "tous_risques": "Tous risques."}
    second = {"aucun": "Non, je suis le seul conducteur.", "conjoint": "Mon conjoint la conduit aussi.",
              "enfant_jeune_conducteur": "Mon fils, jeune conducteur, la conduit aussi.", "parent": "Mon père la conduit aussi."}
    acq = {"comptant": "Je l'ai payée comptant.", "credit": "Je l'ai achetée à crédit.", "LOA_LLD": "Elle est en LOA."}
    n = int(p.sinistres_responsables_36_mois)
    if concept == "garanties":
        voulu = []
        if p.formule_souhaitee == "tous_risques":
            voulu.append("une couverture tous risques")
        elif p.formule_souhaitee == "tiers_plus":
            voulu.append("le vol et le bris de glace")
        if p.option_assistance_0km:
            voulu.append("l'assistance 0 km")
        if p.option_vehicule_remplacement:
            voulu.append("un véhicule de remplacement")
        return ("Surtout " + " et ".join(voulu) + ".") if voulu else "Rien de particulier, le minimum obligatoire me suffit."
    pays_permis = p.pays_permis
    return {
        "type_vehicule": "Une voiture.",
        "vehicule": f"C'est une {p.marque_modele}.",
        "annee_vehicule": f"Elle a été mise en circulation en {int(p.annee_mise_en_circulation)}.",
        "energie": f"Elle roule à l'{p.energie}." if p.energie in ("essence", "electrique") else f"C'est une {p.energie}.",
        "puissance": f"Elle fait {int(p.puissance_ch)} chevaux.",
        "prix_neuf": f"Elle valait environ {nombre(p.prix_neuf_eur)} euros neuve.",
        "acquisition": acq.get(p.mode_acquisition, "Je l'ai payée comptant."),
        "age": f"J'ai {int(p.age)} ans.",
        "commune": f"J'habite à {p.commune}.",
        "pays": "En France.",
        "type_zone": z.get(p.type_zone, "C'est une ville moyenne."),
        "type_permis": "Le permis B.",
        "annees_permis": "J'ai le permis depuis moins d'un an." if int(p.annees_permis) == 0 else f"J'ai le permis depuis {int(p.annees_permis)} an{'s' if int(p.annees_permis) > 1 else ''}.",
        "pays_permis": "Je l'ai passé en France." if pays_permis == "France" else f"Je l'ai passé au {pays_permis}." if pays_permis in ("Maroc", "Portugal") else f"Je l'ai passé en {pays_permis}.",
        "deja_assure": (f"Oui, je suis assuré depuis {int(p.annees_assure_total)} ans." if p.deja_assure else "Non, je n'ai jamais été assuré."),
        "conduite_accompagnee": "Oui, j'ai fait la conduite accompagnée." if p.conduite_accompagnee else "Non, pas de conduite accompagnée.",
        "bonus_malus": f"Mon coefficient bonus-malus est de {p.coefficient_bonus_malus:.2f}.".replace(".", ",", 1),
        "annees_sans_sinistre": f"{int(p.annees_sans_sinistre)} ans sans sinistre responsable.",
        "sinistres": "Aucun sinistre responsable ces 3 dernières années." if n == 0 else f"{n} sinistre{'s' if n > 1 else ''} responsable{'s' if n > 1 else ''} ces 3 dernières années.",
        "resiliation": "Non, je n'ai jamais été résilié." if p.resiliation_assureur == "non" else "Oui, mon assureur a résilié mon contrat.",
        "conducteur_secondaire": second.get(p.conducteur_secondaire, "Non, je suis le seul conducteur."),
        "usage": usage.get(p.usage, "Usage privé."),
        "km": f"Environ {nombre(p.kilometrage_annuel)} km par an.",
        "stationnement": park.get(p.stationnement_nuit, "Dans la rue."),
        "formule": formule.get(p.formule_souhaitee, "Tous risques."),
        "franchise": f"Une franchise {p.franchise_souhaitee}.",
        "budget": f"{nombre(p.budget_mensuel_max_eur)} euros par mois maximum.",
        "critere": "Le prix.",
    }[concept]


# --------------------------------------------------------------------------- reconnaître la question posée
MOTS = {
    "type_vehicule": r"voiture ou (une )?moto|auto ou moto|type de vehicule",
    "vehicule": r"marque|modele|quel vehicule|quelle voiture",
    "annee_vehicule": r"mise en circulation|annee (du|de votre|de la) (vehicule|voiture)|de quelle annee",
    "energie": r"energie|essence|diesel|carburant|motorisation|electrique",
    "puissance": r"puissance|chevaux|\bch\b",
    "prix_neuf": r"prix (a )?neuf|valeur a neuf|prix d.achat",
    "acquisition": r"comptant|credit|\bloa\b|financ",
    "age": r"\bage\b|quel age|date de naissance",
    "commune": r"commune|ville|habitez",
    "pays": r"quel pays habitez|pays de residence",
    "type_zone": r"rural|periurbain|type de zone|metropole",
    "type_permis": r"type de permis|quel permis",
    "annees_permis": r"permis depuis|depuis (combien|quand)[^?]*permis|anciennete[^?]*permis|annees de permis|annees[^?]*permis",
    "pays_permis": r"(ou|quel pays|dans quel pays)[^?]*(obtenu|passe)[^?]*permis|pays[^?]*permis|permis[^?]*(union europeenne|hors ue|hors de l)",
    "deja_assure": r"deja (ete )?assure|avez-vous (deja )?ete assure|assure auparavant",
    "conduite_accompagnee": r"conduite accompagnee|\baac\b",
    "bonus_malus": r"bonus|malus|\bcrm\b|coefficient",
    "annees_sans_sinistre": r"(annees|ans)[^?]*sans sinistre",
    "sinistres": r"sinistre|accident",
    "resiliation": r"resili",
    "conducteur_secondaire": r"seul conducteur|autres? conducteurs?|qui d.autre|conducteur secondaire",
    "usage": r"usage|utilisez",
    "km": r"kilometr|\bkm\b",
    "stationnement": r"gare|stationn|la nuit",
    "formule": r"formule|couverture|tous risques",
    "franchise": r"franchise",
    "garanties": r"garanties|priorit|important",
    "budget": r"budget|par mois",
}


def questions(texte: str) -> list[str]:
    """Les phrases interrogatives du message (le client répond à ce qui est demandé, pas au reste)."""
    t = simple(texte)
    morceaux = [m for m in re.split(r"(?<=[.!?])\s+|\n", t) if "?" in m]
    return morceaux or [t]


def concepts_demandes(texte: str, candidats: list[str]) -> list[str]:
    """Concepts parmi `candidats` que le message demande, dans l'ordre où ils apparaissent."""
    trouves = []
    for q in questions(texte):
        positions = []
        for c in candidats:
            m = re.search(MOTS[c], q)
            if m:
                positions.append((m.start(), c))
        for _, c in sorted(positions):
            if c not in trouves:
                trouves.append(c)
    # un concept plus précis l'emporte : « années sans sinistre » n'est pas « sinistres »
    if "annees_sans_sinistre" in trouves and "sinistres" in trouves:
        trouves.remove("sinistres")
    texte_q = " ".join(questions(texte))
    if "formule" in trouves and "garanties" in trouves:
        # « quelles garanties … (vol, bris de glace, tous risques) » demande les garanties, pas la formule
        trouves.remove("formule" if re.search(r"garanti", texte_q) and not re.search(r"formule|couverture", texte_q) else "garanties")
    if "type_zone" in trouves and "commune" in trouves:
        trouves.remove("commune")
    if "pays_permis" in trouves and "pays" in trouves:
        trouves.remove("pays")
    if "annees_permis" in trouves and "pays_permis" in trouves and not re.search(r"depuis|combien|annees", " ".join(questions(texte))):
        trouves.remove("annees_permis")
    return trouves


CONFIRMATION = r"(souhaitez|voulez|desirez)-vous que je|puis-je (lancer|chercher|rechercher)|je lance la recherche \?|on lance"


def est_confirmation(texte: str) -> bool:
    return bool(re.search(CONFIRMATION, simple(texte)))


# --------------------------------------------------------------------------- notation sur les champs communs
COMMUNS = ["age", "annees_permis", "pays_permis", "sinistres", "vehicule", "annee_vehicule", "usage", "km",
           "stationnement", "commune", "formule", "franchise", "budget"]

# concept du client -> champ commun noté
NOTE = {c: c for c in COMMUNS}


def classe_pays(v) -> str | None:
    if v is None:
        return None
    s = simple(v)
    if s in ("france", "ue", "hors_ue"):
        return s
    if "hors" in s:
        return "hors_ue"
    if s in ("union europeenne", "europe", "ue"):
        return "ue"
    if s == "france":
        return "france"
    return "ue" if any(simple(x) == s for x in UE) else "hors_ue"


def formule_commune(v):
    if v is None:
        return None
    s = simple(v).replace(" ", "_")
    return {"tiers_plus": "intermediaire", "intermediaire": "intermediaire", "tiers": "tiers", "tous_risques": "tous_risques"}.get(s, s)


def verite(p) -> dict:
    return {"age": int(p.age), "annees_permis": int(p.annees_permis), "pays_permis": classe_pays(p.pays_permis),
            "sinistres": int(p.sinistres_responsables_36_mois), "vehicule": p.marque_modele,
            "annee_vehicule": int(p.annee_mise_en_circulation),
            "usage": {"vtc": "professionnel"}.get(p.usage, p.usage), "km": int(p.kilometrage_annuel),
            "stationnement": p.stationnement_nuit, "commune": p.commune, "formule": formule_commune(p.formule_souhaitee),
            "franchise": p.franchise_souhaitee, "budget": float(p.budget_mensuel_max_eur)}


def depuis_adam(f: dict) -> dict:
    return {"age": f.get("age"), "annees_permis": f.get("annees_permis"), "pays_permis": classe_pays(f.get("pays_permis")),
            "sinistres": f.get("sinistres_responsables_36_mois"), "vehicule": f.get("marque_modele"),
            "annee_vehicule": f.get("annee_mise_en_circulation"), "usage": f.get("usage"), "km": f.get("kilometrage_annuel"),
            "stationnement": f.get("stationnement_nuit"), "commune": f.get("commune"),
            "formule": formule_commune(f.get("formule_souhaitee")), "franchise": f.get("franchise_souhaitee"),
            "budget": f.get("budget_mensuel_max")}


def depuis_ismail(f: dict) -> dict:
    usage = {"domicile_travail": "prive_trajet_travail"}
    park = {"garage": "garage_ferme", "parking": "parking_prive", "rue": "voie_publique"}
    return {"age": f.get("age"), "annees_permis": f.get("licenseYears"), "pays_permis": classe_pays(f.get("licenceOrigin")),
            "sinistres": f.get("atFaultClaims"), "vehicule": f.get("vehicle"), "annee_vehicule": f.get("vehicleYear"),
            "usage": usage.get(f.get("usage"), f.get("usage")), "km": f.get("annualKm"),
            "stationnement": park.get(f.get("parking"), f.get("parking")), "commune": f.get("city"),
            "formule": formule_commune(f.get("coverageLevel")), "franchise": f.get("deductible"),
            "budget": f.get("maxMonthlyBudget")}


def mots_modele(v) -> set:
    return {w for w in re.findall(r"[a-z0-9]+", simple(v)) if w not in ("de", "la", "le")}


def egal(champ: str, attendu, obtenu) -> bool:
    if obtenu in (None, "", []):
        return False
    if champ == "vehicule":
        a, b = mots_modele(attendu), mots_modele(obtenu)
        return len(a & b) >= min(2, len(a))
    if champ == "commune":
        return simple(attendu).replace("-", " ").strip() == simple(obtenu).replace("-", " ").strip()
    if champ in ("age", "annees_permis", "sinistres", "annee_vehicule", "km", "budget"):
        try:
            return abs(float(attendu) - float(obtenu)) <= max(1e-6, 0.01 * abs(float(attendu)))
        except (TypeError, ValueError):
            return False
    return simple(attendu) == simple(obtenu)


def noter(attendu: dict, obtenu: dict, dits: set) -> dict:
    """Pour chaque champ commun : juste / faux / oublié (dit mais vide) / inventé (pas dit mais rempli) / vide (ni l'un ni l'autre)."""
    res = {}
    for c in COMMUNS:
        rempli = obtenu.get(c) not in (None, "", [])
        if c in dits:
            res[c] = "juste" if egal(c, attendu[c], obtenu.get(c)) else ("faux" if rempli else "oublie")
        else:
            res[c] = "invente" if rempli else "vide"
    return res
