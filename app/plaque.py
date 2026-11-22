"""
Recherche d'un véhicule à partir de sa plaque d'immatriculation (API RegCheck, France).

Le client tape sa plaque, on récupère la marque, le modèle, l'année et l'énergie, et on remplit la fiche.
Rien n'est enregistré : la plaque sert seulement à cette recherche.

Réglage dans .env :
    REGCHECK_USER=votre_nom_utilisateur      (compte gratuit sur regcheck.org.uk)
Plaque de test gratuite (ne consomme pas de crédit) : EG258MA
"""
from __future__ import annotations

import html
import json
import os
import re
import urllib.parse
import urllib.request

AUTO = os.environ.get("REGCHECK_URL", "https://www.regcheck.org.uk/api/reg.asmx") + "/CheckFrance"
MOTO = os.environ.get("REGCHECK_URL_MOTO", "https://www.immatriculationapi.com/api/bespokeapi.asmx") + "/CheckMotorBikeFrance"

# plaques françaises : SIV (AB-123-CD, depuis 2009) et ancien format FNI (1234 AB 72)
SIV = re.compile(r"\b([A-HJ-NP-TV-Z]{2})[\s-]?(\d{3})[\s-]?([A-HJ-NP-TV-Z]{2})\b", re.I)
FNI = re.compile(r"\b(\d{1,4})[\s-]?([A-Z]{1,3})[\s-]?(\d{2}|2A|2B|97\d)\b", re.I)

SIGLES = {"BMW", "DS", "MG", "KTM", "BYD", "VW", "GT", "GTI", "SUV", "TDI", "RS"}


def disponible() -> bool:
    return bool(os.environ.get("REGCHECK_USER"))


def trouver_plaque(texte: str) -> str | None:
    """Plaque écrite dans un texte, normalisée (AB-123-CD), ou None."""
    m = SIV.search(texte)
    if m:
        return "-".join(g.upper() for g in m.groups())
    m = FNI.search(texte)
    if m and not re.fullmatch(r"\d+", m.group(0).replace(" ", "")):
        return " ".join(g.upper() for g in m.groups())
    return None


def joli(nom: str) -> str:
    """RENAULT SCÉNIC III -> Renault Scénic III (sigles et chiffres romains gardés)."""
    mots = []
    for w in nom.split():
        if w.upper() in SIGLES or re.fullmatch(r"[IVX]+", w) or re.search(r"\d", w):
            mots.append(w.upper())
        else:
            mots.append("-".join(p.capitalize() for p in w.lower().split("-")))
    return " ".join(mots)


def energie(donnees: dict) -> str | None:
    ext = donnees.get("ExtendedData") or {}
    if str(ext.get("electrique", "")).upper() in ("O", "OUI", "Y"):
        return "electrique"
    f = str((donnees.get("FuelType") or {}).get("CurrentTextValue") or "").upper()
    if "HYBRID" in f:
        return "hybride"
    if "ELECTR" in f:
        return "electrique"
    if "DIESEL" in f or "GAZOLE" in f:
        return "diesel"
    if "ESSENCE" in f or "PETROL" in f or "GASOLINE" in f:
        return "essence"
    return None


def _appel(url: str, plaque: str) -> dict | None:
    params = urllib.parse.urlencode({"RegistrationNumber": plaque.replace("-", "").replace(" ", ""),
                                     "username": os.environ["REGCHECK_USER"]})
    with urllib.request.urlopen(f"{url}?{params}", timeout=15) as rep:
        xml = rep.read().decode("utf-8", "replace")
    m = re.search(r"<vehicleJson>(.*?)</vehicleJson>", xml, re.S)
    if not m or not m.group(1).strip():
        return None
    return json.loads(html.unescape(m.group(1)))


def chercher(plaque: str, type_vehicule: str | None = None) -> dict:
    """Retourne {"ok": True, "vehicule": {...}, "fiche": {...}} ou {"ok": False, "raison": "..."}."""
    if not disponible():
        return {"ok": False, "raison": "indisponible"}
    urls = [MOTO, AUTO] if type_vehicule == "moto" else [AUTO]
    donnees = None
    for url in urls:
        try:
            donnees = _appel(url, plaque)
        except Exception:  # plaque inconnue, crédits épuisés ou service injoignable : l'API répond par une erreur
            donnees = None
        if donnees:
            break
    if not donnees:
        return {"ok": False, "raison": "introuvable"}
    marque = (donnees.get("CarMake") or donnees.get("MakeDescription") or {}).get("CurrentTextValue") or ""
    modele = (donnees.get("CarModel") or donnees.get("ModelDescription") or {}).get("CurrentTextValue") or ""
    nom = joli(f"{marque} {modele}".strip() or donnees.get("Description", ""))
    ext = donnees.get("ExtendedData") or {}
    genre = str(ext.get("genre", "")).upper()
    type_v = "moto" if genre.startswith("MT") or genre in ("CL", "MOTO") or url == MOTO else "auto"
    try:
        annee = int(str(donnees.get("RegistrationYear") or ext.get("anneeSortie"))[:4])
    except ValueError:
        annee = None
    try:
        cv = int(ext.get("puissance") or (donnees.get("EngineSize") or {}).get("CurrentTextValue") or 0) or None
    except ValueError:
        cv = None
    fiche = {"type_vehicule": type_v, "marque_modele": nom or None, "annee_mise_en_circulation": annee,
             "energie": energie(donnees), "puissance_fiscale": cv if type_v == "auto" else None}
    vehicule = {"nom": nom, "annee": annee, "energie": fiche["energie"],
                "carrosserie": joli((donnees.get("BodyStyle") or {}).get("CurrentTextValue") or "") or None,
                "version": (ext.get("libVersion") or "").strip() or None,
                "places": ext.get("nbPlace") or None,
                "image": donnees.get("ImageUrl") or None}
    return {"ok": True, "vehicule": vehicule, "fiche": {k: v for k, v in fiche.items() if v is not None}}
