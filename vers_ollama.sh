#!/usr/bin/env bash
# Met le modèle fine-tuné dans Ollama, pour l'utiliser comme celui d'Ismail (même API, même évaluation).
# Usage :  bash vers_ollama.sh <dossier_modele_fusionne> <nom_ollama>
# Ollama importe directement les safetensors (architecture Qwen2) : pas besoin de llama.cpp ni de Python.
set -euo pipefail

SOURCE="$(cd "${1:?dossier du modèle fusionné}" && pwd)"
NOM="${2:-assuria-profil}"
TRAVAIL="$(dirname "$SOURCE")"

# Gabarit ChatML de Qwen2.5, le même que pendant l'entraînement (le message système est toujours fourni)
cat > "$TRAVAIL/Modelfile" <<EOF
FROM $SOURCE
TEMPLATE """{{- range .Messages }}<|im_start|>{{ .Role }}
{{ .Content }}<|im_end|>
{{ end }}<|im_start|>assistant
"""
PARAMETER stop "<|im_end|>"
PARAMETER stop "<|endoftext|>"
PARAMETER temperature 0
PARAMETER num_ctx 4096
EOF

# q8_0 : presque sans perte, 2 fois plus petit que le modèle d'origine
ollama create "$NOM" -f "$TRAVAIL/Modelfile" --quantize q8_0
echo "Prêt : ollama run $NOM"
