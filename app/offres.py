"""
Recherche des offres et estimation du prix, à partir des données du projet.

Les données viennent de la base Neon (schéma objectif1) si NEON_DATABASE_URL est défini, sinon des fichiers du dépôt
assuria-projet cloné à côté (../assuria-projet/data/objectif1).

Prix estimé = prix de base de l'offre pour la formule × coefficients du profil.
Les coefficients sont ceux du simulateur du projet (médianes de la table offres_prix_profils) ; le bonus-malus suit la
règle française (×0,95 par an sans sinistre, ×1,25 par sinistre responsable, entre 0,50 et 3,50).
C'est une estimation indicative, pas un devis.
"""
from __future__ import annotations

import os
import re
from functools import lru_cache
from pathlib import Path

import pandas as pd
from datetime import date

ANNEE = date.today().year
LOCAL = Path(os.environ.get("ASSURIA_DATA", Path(__file__).resolve().parents[2] / "assuria-projet" / "data" / "objectif1"))

# coefficients du simulateur (auto / moto)
COEF_USAGE = {"prive": 1.0, "loisirs": 1.0, "prive_trajet_travail": 1.05, "professionnel": 1.2, "vtc": 1.2}
COEF_STATIONNEMENT = {"garage_ferme": 0.93, "parking_prive": 1.0, "voie_publique": 1.08}
COEF_ZONE = {"auto": {"rural": 0.822, "periurbain": 0.929, "urbain": 1.0, "grande_ville": 1.3, "metropole": 1.543},
             "moto": {"rural": 0.672, "periurbain": 0.754, "urbain": 1.0, "grande_ville": 1.306, "metropole": 1.306}}
AGE = {"auto": [(30, 1.208), (45, 1.004), (60, 1.0), (70, 0.886), (200, 1.09)],
       "moto": [(30, 1.818), (45, 1.247), (60, 1.0), (70, 0.921), (200, 0.921)]}
KM = [(5000, 0.85), (10000, 0.93), (15000, 1.0), (20000, 1.07), (30000, 1.15), (10 ** 9, 1.25)]
JEUNE_CONDUCTEUR = {0: 2.0, 1: 1.5, 2: 1.25}
# avec la conduite accompagnée, la surprime est réduite de moitié la 1re année et s'arrête après 2 ans
JEUNE_CONDUCTEUR_AAC = {0: 1.5, 1: 1.25, 2: 1.0}
# coefficients appris sur des sinistres réels (simulation/coefficients.json du projet : freMTPL2 pour l'auto,
# swmotorcycle pour la moto) et règles du marché
ENERGIE = {"auto": {"diesel": 1.1932}, "moto": {}}
AGE_VEHICULE = {"auto": [(0, 0.9927), (2, 0.9749), (5, 1.0), (10, 1.0391), (15, 0.9094), (10 ** 4, 0.7228)],
                "moto": [(2, 1.3024), (5, 1.0), (10, 0.8766), (15, 0.7297), (10 ** 4, 0.5776)]}
PUISSANCE_AUTO = [(4, 0.9288), (5, 1.0), (6, 1.0852), (7, 1.0602), (8, 1.0268), (10, 1.2223), (10 ** 4, 1.313)]
FRANCHISE = {"basse": 1.1, "standard": 1.0, "haute": 0.88}
PERMIS_HORS_UE = 1.1
UE = {"France", "Belgique", "Luxembourg", "Allemagne", "Pays-Bas", "Italie", "Espagne", "Portugal", "Irlande", "Autriche",
      "Suède", "Danemark", "Finlande", "Pologne", "Tchéquie", "République tchèque", "Slovaquie", "Hongrie", "Roumanie",
      "Bulgarie", "Grèce", "Croatie", "Slovénie", "Estonie", "Lettonie", "Lituanie", "Malte", "Chypre",
      "Norvège", "Islande", "Liechtenstein", "Suisse", "Union européenne"}  # Union européenne + EEE + Suisse : permis reconnus sans surprime
# valeur d'une unité de devise en euros (approximatif, comme pour les profils)
TAUX_EUR = {"EUR": 1.0, "GBP": 1.17, "CHF": 1.07, "PLN": 0.235, "SEK": 0.087, "NOK": 0.085, "DKK": 0.134, "CZK": 0.04, "RON": 0.20}


# --------------------------------------------------------------------------- accès aux données
def source() -> str:
    return "Neon" if os.environ.get("NEON_DATABASE_URL") else ("fichiers locaux" if LOCAL.exists() else "aucune")


@lru_cache(maxsize=1)
def _moteur():
    from sqlalchemy import create_engine
    url = os.environ["NEON_DATABASE_URL"].replace("postgres://", "postgresql://", 1)
    return create_engine(url.replace("postgresql://", "postgresql+psycopg2://", 1), pool_pre_ping=True)


@lru_cache(maxsize=8)
def table(nom: str, colonnes: str = "*") -> pd.DataFrame:
    if os.environ.get("NEON_DATABASE_URL"):
        return pd.read_sql(f'SELECT {colonnes} FROM objectif1."{nom}"', _moteur())
    fichiers = {"prix_base": "simulation/prix_base.parquet", "offres_europe": "offres_europe/offres_europe.parquet",
                "profils": "profils/profils.parquet", "reputation_assureurs": "reputation/reputation_assureurs.csv"}
    f = LOCAL / fichiers[nom]
    df = pd.read_csv(f) if f.suffix == ".csv" else pd.read_parquet(f)
    return df if colonnes == "*" else df[[c.strip() for c in colonnes.split(",")]]


def zone_de(commune: str | None) -> str | None:
    """Type de zone de la commune (rural, urbain, métropole…), d'après les villes connues de la table profils."""
    if not commune:
        return None
    villes = table("profils", "commune, type_zone").drop_duplicates("commune")
    m = trouver_ville(villes, commune)
    return None if m.empty else m.type_zone.iloc[0]


def trouver_ville(villes: pd.DataFrame, commune: str) -> pd.DataFrame:
    c = commune.strip().lower()
    m = villes[villes.commune.str.lower() == c]
    if m.empty:
        m = villes[villes.commune.str.lower().str.startswith(c + " ")]
    if m.empty and " " in c:  # « paris 13e » -> une ville qui commence par « paris »
        m = villes[villes.commune.str.lower().str.startswith(c.split(" ")[0] + " ")]
    return m


def deduire(quoi: str, valeur: str):
    """Déductions sûres depuis la base : le pays d'une ville connue, le type (auto/moto) d'un modèle connu."""
    try:
        if quoi == "pays":
            villes = table("profils", "commune, pays_residence").drop_duplicates("commune")
            m = trouver_ville(villes, valeur)
            return None if m.empty else m.pays_residence.iloc[0]
        if quoi == "type":
            modeles = table("profils", "marque_modele, type_vehicule").drop_duplicates("marque_modele")
            v = valeur.strip().lower()
            m = modeles[modeles.marque_modele.str.lower().apply(lambda x: x == v or x.endswith(" " + v) or v in x)]
            return None if m.empty else m.type_vehicule.iloc[0]
    except Exception:
        return None
    return None


# --------------------------------------------------------------------------- tarification
def coef_bonus_malus(fiche: dict) -> float:
    bm = str(fiche.get("bonus_malus") or "")
    trouve = re.search(r"CRM\s*([0-9]+[.,][0-9]+)", bm, re.I)
    if trouve:
        return float(trouve.group(1).replace(",", "."))
    if fiche.get("deja_assure") is False:
        return 1.0
    annees = fiche.get("annees_assure_total") or 0
    sin = fiche.get("sinistres_responsables_36_mois") or 0
    return round(min(3.5, max(0.5, 0.95 ** max(0, annees - sin) * 1.25 ** sin)), 2)


def coefficients(fiche: dict, type_v: str) -> dict:
    age = min(fiche.get("age") or 40, 199)
    km = min(fiche.get("kilometrage_annuel") or 12000, 10 ** 8)
    zone = zone_de(fiche.get("commune"))
    novice = JEUNE_CONDUCTEUR_AAC if fiche.get("conduite_accompagnee") else JEUNE_CONDUCTEUR
    annee = fiche.get("annee_mise_en_circulation")
    c = {
        "âge": next(v for borne, v in AGE[type_v] if age <= borne),
        "bonus-malus": coef_bonus_malus(fiche),
        # conducteur novice : moins de 3 ans de permis, ou jamais assuré (premier contrat)
        # conducteur novice : jamais assuré (1re année de surprime) ou moins de 3 ans de permis
        "jeune conducteur": novice[0] if fiche.get("deja_assure") is False else novice.get(fiche.get("annees_permis"), 1.0),
        "usage": COEF_USAGE.get(fiche.get("usage"), 1.0),
        "kilométrage": next(v for borne, v in KM if km <= borne),
        "stationnement": COEF_STATIONNEMENT.get(fiche.get("stationnement_nuit"), 1.0),
        "zone": COEF_ZONE[type_v].get(zone, 1.0),
        "énergie": ENERGIE[type_v].get(fiche.get("energie"), 1.0),
        "âge du véhicule": next(v for borne, v in AGE_VEHICULE[type_v] if (ANNEE - annee if annee else 4) <= borne),
        "puissance": next(v for borne, v in PUISSANCE_AUTO if fiche["puissance_fiscale"] <= borne)
                     if type_v == "auto" and fiche.get("puissance_fiscale") else 1.0,
        "franchise": FRANCHISE.get(fiche.get("franchise_souhaitee"), 1.0) if fiche.get("formule_souhaitee") != "tiers" else 1.0,
        "permis étranger": PERMIS_HORS_UE if fiche.get("pays_permis") and fiche["pays_permis"] not in UE else 1.0,
    }
    return {k: round(v, 3) for k, v in c.items()}


# la formule « intermédiaire » s'appelle tiers_plus dans la fiche et intermediaire dans les tables d'offres
FORMULE_TABLE = {"tiers_plus": "intermediaire"}

# garanties comprises dans chaque formule (les autres garanties du document sont des options d'autres formules)
DANS_FORMULE = {
    "tiers": {"responsabilite_civile", "assistance", "protection_conducteur", "protection_juridique"},
    "intermediaire": {"responsabilite_civile", "assistance", "protection_conducteur", "protection_juridique", "vol", "incendie",
                   "bris_de_glace", "catastrophes_naturelles"},
}

# --------------------------------------------------------------------------- réputation
def reputation() -> dict:
    """{(pays, assureur): {note, avis, plateforme, source}} ; vide si la table n'est pas encore disponible."""
    try:
        df = table("reputation_assureurs")
    except Exception:
        return {}
    return {(r.pays, r.assureur): {"note": float(r.note_sur_5), "avis": int(r.nombre_avis), "plateforme": r.plateforme,
                                   "source": r.source} for r in df.itertuples()}


def score_reputation(o: dict) -> float:
    """Note corrigée par le nombre d'avis (moyenne bayésienne) : 4,8 sur 10 avis compte moins que 4,3 sur 5 000."""
    if not o.get("note_clients"):
        return 0.0
    m, moyenne = 300, 3.3
    return (o["nombre_avis"] * o["note_clients"] + m * moyenne) / (o["nombre_avis"] + m)


ECART_PROCHE = 0.05  # deux prix à moins de 5 % l'un de l'autre sont « proches » : les avis départagent


def classer(offres: list[dict], critere: str = "prix") -> list[dict]:
    """Classe les offres selon le critère du client ; à prix proches, la meilleure réputation passe devant."""
    for o in offres:
        o["departagee"] = False
    if critere == "reputation":
        return sorted(offres, key=lambda o: (o["dans_le_budget"] is False, -score_reputation(o), o["prix_mensuel_estime_eur"]))
    if critere == "equilibre":  # prix 50 %, garanties 25 %, avis 25 %
        pmin = min(o["prix_mensuel_estime_eur"] for o in offres)
        gmax = max(len(o["garanties"]) for o in offres) or 1
        score = lambda o: (0.5 * pmin / o["prix_mensuel_estime_eur"] + 0.25 * len(o["garanties"]) / gmax
                           + 0.25 * score_reputation(o) / 5)
        return sorted(offres, key=lambda o: (o["dans_le_budget"] is False, -score(o)))
    if critere == "garanties":
        return sorted(offres, key=lambda o: (o["dans_le_budget"] is False, -len(o["garanties"]), o["prix_mensuel_estime_eur"]))
    liste = sorted(offres, key=lambda o: (o["dans_le_budget"] is False, o["prix_mensuel_estime_eur"]))
    # groupes de prix proches (à moins de 5 % de la moins chère du groupe) : dans un groupe, la meilleure réputation d'abord
    resultat, groupe = [], []
    for o in liste + [None]:
        if groupe and (o is None or o["dans_le_budget"] != groupe[0]["dans_le_budget"]
                       or o["prix_mensuel_estime_eur"] > groupe[0]["prix_mensuel_estime_eur"] * (1 + ECART_PROCHE)):
            trie = sorted(groupe, key=lambda x: -score_reputation(x))
            if trie[0] is not groupe[0]:
                trie[0]["departagee"] = True
            resultat += trie
            groupe = []
        if o is not None:
            groupe.append(o)
    return resultat


# --------------------------------------------------------------------------- recherche
def chercher(fiche: dict, n: int | None = None) -> tuple[list[dict], dict, str]:
    """Retourne (offres classées, coefficients appliqués, message éventuel)."""
    type_v = fiche.get("type_vehicule") or "auto"
    pays, formule = fiche.get("pays_residence"), fiche.get("formule_souhaitee") or "tous_risques"
    formule = FORMULE_TABLE.get(formule, formule)
    if (fiche.get("sinistres_responsables_36_mois") or 0) >= 3:
        return [], {}, "Avec 3 sinistres responsables ou plus en 3 ans, il faut s'adresser à un assureur spécialisé."
    prix = table("prix_base")
    choix = prix[(prix.pays == pays) & (prix.type == type_v) & (prix.formule == formule)]
    message = ""
    if choix.empty and pays != "France":
        choix = prix[(prix.pays == "France") & (prix.type == type_v) & (prix.formule == formule)]
        message = (f"Pas d'offre {type_v} en base pour « {pays} » : offres françaises affichées à titre d'exemple." if pays else
                   "Pays non communiqué : offres françaises affichées à titre d'exemple.")
    specialise = choix.produit.str.contains("maluss", case=False, na=False)  # produits pour malussés / résiliés
    resilie = fiche.get("resiliation_assureur") not in (None, "non")
    if resilie:
        if not specialise.any():
            return [], {}, ("Votre contrat a été résilié par un assureur : il faut passer par une offre spécialisée pour "
                            "conducteurs résiliés, que nous n'avons pas en base pour ce pays.")
        choix = choix[specialise]
        message = "Contrat résilié par un assureur : seules les offres pour conducteurs résiliés sont proposées."
    elif coef_bonus_malus(fiche) <= 1.0:  # les produits réservés aux conducteurs malussés ne concernent pas ce client
        choix = choix[~specialise]
    if fiche.get("conducteur_secondaire") == "enfant_jeune_conducteur":
        message = (message + " " if message else "") + ("Un jeune conducteur déclaré en conducteur secondaire augmente "
                   "généralement la cotisation : l'estimation ne l'inclut pas, l'assureur l'ajoutera au devis.")
    if fiche.get("mode_acquisition") in ("credit", "LOA_LLD") and fiche.get("formule_souhaitee") == "tiers":
        message = (message + " " if message else "") + ("Véhicule financé à crédit ou en LOA/LLD : l'organisme de "
                   "financement exige souvent une assurance tous risques, vérifiez votre contrat.")
    coefs = coefficients(fiche, type_v)
    total = 1.0
    for v in coefs.values():
        total *= v
    offres = table("offres_europe").set_index("id_offre")
    budget_eur = None
    if fiche.get("budget_mensuel_max"):
        budget_eur = fiche["budget_mensuel_max"] * TAUX_EUR.get(fiche.get("devise_budget") or "EUR", 1.0)
    notes = reputation()
    resultats = []
    for r in choix.itertuples():
        mensuel = round(r.prix_base_annuel * total / 12, 2)
        o = offres.loc[r.id_offre] if r.id_offre in offres.index else None
        permises = DANS_FORMULE.get(r.formule)
        garanties = [g[2:].replace("_", " ") for g in offres.columns
                     if g.startswith("g_") and o is not None and bool(o[g]) and (permises is None or g[2:] in permises)]
        resultats.append({
            "assureur": r.assureur, "produit": r.produit, "formule": "tiers_plus" if r.formule == "intermediaire" else r.formule, "prix_mensuel_estime_eur": mensuel,
            "dans_le_budget": None if budget_eur is None else mensuel <= budget_eur,
            "garanties": garanties,
            "franchise_min_eur": None if o is None or pd.isna(o.franchise_min_eur) else float(o.franchise_min_eur),
            "lien": r.url_offre,
            "note_clients": (notes.get((r.pays, r.assureur)) or {}).get("note"),
            "nombre_avis": (notes.get((r.pays, r.assureur)) or {}).get("avis"),
            "plateforme_avis": (notes.get((r.pays, r.assureur)) or {}).get("plateforme"),
            "source_avis": (notes.get((r.pays, r.assureur)) or {}).get("source"),
        })
    # une offre par assureur, les moins chères d'abord, celles dans le budget en premier
    vues, classees = set(), []
    for r in sorted(resultats, key=lambda x: (x["dans_le_budget"] is False, x["prix_mensuel_estime_eur"])):
        if r["assureur"] not in vues:
            vues.add(r["assureur"])
            classees.append(r)
    return classees[:n] if n else classees, coefs, message
