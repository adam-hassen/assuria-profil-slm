# Benchmark des deux approches de l'objectif 1

- **A (Adam)** : petit modèle fine-tuné `assuria-profil` + code + `qwen3:8b` (ce repo, `app/`)
- **B (Ismail)** : agent `qwen3:8b` avec outils, framework Mastra (`ismailAchibat/assuria`)

| Test | Ce qu'il mesure | Script |
|---|---|---|
| 1. Extraction | 30 conversations écrites à la main, données d'un coup : qui remplit le mieux la fiche ? | `bench_extraction.py` |
| 2. Conversations | un client simulé (vrai profil France, mêmes phrases pour les deux) parle à chaque système jusqu'aux offres : temps, tours, fiche finale | `bench_conversations.py` |

La notation porte sur les champs que les deux fiches ont en commun (âge, permis, pays du permis, sinistres, véhicule,
année, usage, km, stationnement, commune, formule, franchise, budget) : juste / faux / oublié / inventé.

## Installation (une fois)
```bash
cd ~ && git clone https://github.com/ismailAchibat/assuria ismail-assuria
cd ~/ismail-assuria && npm install
cp ~/assuria-profil-slm/benchmark/ismail/bench-bridge.ts src/scripts/
cp .env.example .env
sed -i '' '/^NEON_PGSQL_CONNECTION_STRING=/d' .env
printf '\nNEON_PGSQL_CONNECTION_STRING=%s\n' "$(grep '^NEON_DATABASE_URL=' ~/assuria-profil-slm/.env | cut -d= -f2-)" >> .env
cd ~ && git clone https://github.com/adam-hassen/assuria-projet   # les profils des clients simulés
```

## Lancer
Terminal 1 (agent d'Ismail) :
```bash
cd ~/ismail-assuria && npx tsx --env-file=.env src/scripts/bench-bridge.ts
```
Terminal 2 :
```bash
cd ~/assuria-profil-slm
python3 evaluer.py --backend ollama --modele qwen3:8b --nom qwen3-8b-ecrit --jeu ecrit
python3 benchmark/bench_extraction.py
python3 benchmark/bench_conversations.py --systeme ismail --n 10
```
Puis arrêter le terminal 1 (Ctrl+C) et y lancer l'app d'Adam, petit modèle seul :
```bash
cd ~/assuria-profil-slm && REGLES=0 uvicorn app.serveur:app
```
Terminal 2 :
```bash
python3 benchmark/bench_conversations.py --systeme adam --n 10
```
Résultats : `benchmark/resultats/`.
