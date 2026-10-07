#!/bin/bash
# One-time setup on CSCS Alps (Clariden): venv on the pytorch uenv, PTB file names, encoder cache.
# Run on a login node from the repo root: bash setup_alps.sh
set -euo pipefail
ROOT=${PARTIAL_ROOT:-/capstor/store/cscs/swissai/a0087/tianyu/partial}
UENV=${UENV:-pytorch/v2.9.1:v2}
mkdir -p "$ROOT"/{hf_cache,checkpoints,logs}

# The partial-order tagger reads data/ptb/{split}.gold.conllu (learning/dataset.py via const.DEP_PATH).
for s in train dev test; do ln -sf "ptb_${s}_3.3.0.sd.clean" "data/ptb/${s}.gold.conllu"; done

export HF_HOME=$ROOT/hf_cache HF_HUB_CACHE=$ROOT/hf_cache/hub
uenv run --view=default "$UENV" -- bash -c '
  [ -d .venv ] || python -m venv --system-site-packages .venv
  .venv/bin/pip install -q -r requirements-alps.txt
  .venv/bin/python -c "
from huggingface_hub import snapshot_download
for m in [\"xlnet-large-cased\", \"bert-base-cased\"]:
    print(snapshot_download(m, allow_patterns=[\"*.json\", \"*.txt\", \"*.model\", \"*.safetensors\", \"pytorch_model.bin\"]))
"'
