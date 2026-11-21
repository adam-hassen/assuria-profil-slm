"""
Interface de démonstration : tu parles comme un client, l'agent te répond et la fiche se remplit en direct.

Lancement (depuis la racine du dépôt) :
    streamlit run app/app.py
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
import agent  # noqa: E402
import offres  # noqa: E402

# connexion Neon : fichier .env à la racine du dépôt (NEON_DATABASE_URL=postgresql://...)
env = Path(__file__).resolve().parent.parent / ".env"
if env.exists():
    for ligne in env.read_text().splitlines():
        if "=" in ligne and not ligne.startswith("#"):
            k, v = ligne.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"'))

st.set_page_config(page_title="AssurIA — Conseiller", layout="wide")

# --------------------------------------------------------------------------- barre latérale : choix des modèles
installes = agent.modeles_installes()
with st.sidebar:
    st.header("Réglages")
    if not installes:
        st.error("Ollama ne répond pas. Lance l'application Ollama, puis recharge la page.")
    def choix(titre, prefere, aide):
        options = installes or [prefere]
        return st.selectbox(titre, options, index=options.index(prefere) if prefere in options else 0, help=aide)
    slm = choix("Petit modèle (remplit la fiche)", "qwen2.5:1.5b",
                "Le modèle de base pour commencer ; plus tard, le modèle fine-tuné (ex. assuria-profil).")
    llm = choix("Gros LLM (discute avec le client)", "qwen3:8b", "Pose les questions et explique la recommandation.")
    st.caption(f"Données des offres : {offres.source()}")
    if st.button("Nouvelle conversation", width="stretch"):
        st.session_state.clear()
        st.rerun()

# --------------------------------------------------------------------------- état de la conversation
if "historique" not in st.session_state:
    st.session_state.historique = [{"role": "assistant", "content": agent.ACCUEIL}]
    st.session_state.fiche = {c: None for c in agent.NOMS}
    st.session_state.journal = []  # ce que le petit modèle a extrait à chaque tour, avec le temps
    st.session_state.offres = None

st.title("AssurIA — Conseiller assurance auto et moto")
gauche, droite = st.columns([3, 2], gap="large")

# --------------------------------------------------------------------------- la fiche (à droite)
def afficher_fiche(zone):
    fiche = st.session_state.fiche
    with zone.container():
        st.subheader("Fiche du client")
        n, total = agent.cases_connues(fiche), len(agent.A_DEMANDER)
        st.progress(n / total, text=f"{n} / {total} informations nécessaires")
        lignes = [{"Case": c, "Valeur": "" if v is None else str(v)} for c, v in fiche.items()]
        st.dataframe(pd.DataFrame(lignes), hide_index=True, width="stretch", height=560)
        if st.session_state.journal:
            j = st.session_state.journal[-1]
            st.caption(f"Dernière extraction : {j['duree']:.1f} s par {slm}")
            with st.expander("Réponse brute du petit modèle (dernier tour)"):
                st.code(j["brut"], language="json")


zone_fiche = droite.empty()
afficher_fiche(zone_fiche)

# --------------------------------------------------------------------------- le chat (à gauche)
with gauche:
    for m in st.session_state.historique:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
    if st.session_state.offres:
        with st.expander("Offres trouvées (détail)", expanded=False):
            st.dataframe(pd.DataFrame(st.session_state.offres), hide_index=True, width="stretch")

message = st.chat_input("Écris ta réponse comme un client…", disabled=not installes)
if message:
    st.session_state.historique.append({"role": "user", "content": message})
    with gauche:
        with st.chat_message("user"):
            st.markdown(message)

        # 1. le petit modèle met la fiche à jour
        with st.spinner("Le petit modèle met la fiche à jour…"):
            try:
                nouvelle, duree, brut = agent.extraire_fiche(slm, st.session_state.historique)
            except Exception as err:
                st.error(f"Le petit modèle n'a pas répondu : {err}")
                st.stop()
        st.session_state.fiche = agent.fusionner(st.session_state.fiche, nouvelle)
        st.session_state.journal.append({"duree": duree, "brut": brut})
        afficher_fiche(zone_fiche)

        # 2. le gros LLM pose la question suivante, ou présente les offres
        suivante = agent.case_suivante(st.session_state.fiche)
        with st.chat_message("assistant"):
            try:
                if suivante:
                    reponse = st.write_stream(agent.question_suivante(llm, st.session_state.fiche,
                                                                      st.session_state.historique, suivante[1]))
                else:
                    with st.spinner("Recherche des offres…"):
                        trouvees, coefs, info = offres.chercher(st.session_state.fiche)
                    st.session_state.offres = trouvees
                    if info:
                        st.info(info)
                    if trouvees:
                        reponse = st.write_stream(agent.recommandation(llm, st.session_state.fiche, trouvees))
                        st.caption("Coefficients appliqués au prix de base : " +
                                   ", ".join(f"{k} ×{v}" for k, v in coefs.items()))
                        st.dataframe(pd.DataFrame(trouvees)[["assureur", "produit", "formule", "prix_mensuel_estime_eur",
                                                              "dans_le_budget", "lien"]], hide_index=True)
                    else:
                        reponse = info or "Je n'ai trouvé aucune offre correspondante dans notre base."
                        st.markdown(reponse)
            except Exception as err:
                st.error(f"Erreur : {err}")
                st.stop()
    st.session_state.historique.append({"role": "assistant", "content": reponse})
