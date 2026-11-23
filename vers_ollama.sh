#!/usr/bin/env bash
# Met le modèle fine-tuné dans Ollama, pour l'utiliser comme celui d'Ismail (même API, même évaluation).
# Usage :  bash vers_ollama.sh <dossier_modele_fusionne> <nom_ollama>
# Étapes : conversion en GGUF avec llama.cpp (dans un Python 3.12 isolé, installé par uv), puis ollama create.
set -euo pipefail

SOURCE="$(cd "${1:?dossier du modèle fusionné}" && pwd)"
NOM="${2:-assuria-profil}"
TRAVAIL="$(dirname "$SOURCE")"

# 1. llama.cpp (le script de conversion)
if [ ! -d "$TRAVAIL/llama.cpp" ]; then
  git clone --depth 1 https://github.com/ggml-org/llama.cpp "$TRAVAIL/llama.cpp"
fi

# 2. Un Python récent isolé (le Python 3.9 de macOS est trop vieux pour llama.cpp)
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
ENV="$TRAVAIL/.venv-gguf"
if [ ! -x "$ENV/bin/python" ]; then
  uv venv --python 3.12 "$ENV"
  uv pip install --python "$ENV/bin/python" -r "$TRAVAIL/llama.cpp/requirements/requirements-convert_hf_to_gguf.txt"
fi

# 3. Conversion en GGUF q8_0 (presque sans perte, 2 fois plus petit)
"$ENV/bin/python" "$TRAVAIL/llama.cpp/convert_hf_to_gguf.py" "$SOURCE" --outtype q8_0 --outfile "$TRAVAIL/$NOM.gguf"

# 4. Gabarit ChatML de Qwen2.5, le même que pendant l'entraînement (le message système est toujours fourni)
cat > "$TRAVAIL/Modelfile" <<EOF
FROM ./$NOM.gguf
TEMPLATE """{{- range .Messages }}<|im_start|>{{ .Role }}
{{ .Content }}<|im_end|>
{{ end }}<|im_start|>assistant
"""
PARAMETER stop "<|im_end|>"
PARAMETER stop "<|endoftext|>"
PARAMETER temperature 0
PARAMETER num_ctx 4096
EOF

(cd "$TRAVAIL" && ollama create "$NOM" -f Modelfile)
echo "Prêt : ollama run $NOM"
