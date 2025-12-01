#!/usr/bin/env bash
set -euo pipefail

# One-click script: download data/model (from HF) and launch NYT training.
# Requires: python with packages from requirements.txt and (optionally) HF_TOKEN env for private downloads.

ROOT_DIR="$(cd -- "$(dirname "$0")/.." && pwd)"
export ROOT_DIR
export HF_HOME="${HF_HOME:-"$ROOT_DIR/.cache/huggingface"}"
mkdir -p "$HF_HOME"

python - <<'PY'
import os
from pathlib import Path
import shutil
from huggingface_hub import snapshot_download

root = Path(os.environ["ROOT_DIR"])
token = os.environ.get("HF_TOKEN")

def ensure_snapshot(repo_id, repo_type, target_dir, allow_patterns=None):
    target_dir = Path(target_dir)
    if target_dir.exists():
        print(f"[skip] {target_dir} already exists")
        return
    print(f"[download] {repo_id} -> {target_dir}")
    snapshot_download(
        repo_id=repo_id,
        repo_type=repo_type,
        local_dir=target_dir,
        local_dir_use_symlinks=False,
        token=token,
        allow_patterns=allow_patterns,
    )

# Download preprocessed NYT data and bert-base-cased weights.
ensure_snapshot(
    repo_id="YF0808/tplinker-nyt-data4bert",
    repo_type="dataset",
    target_dir=root / "data4bert" / "nyt",
)

nyt_target = root / "data4bert" / "nyt"
nested_data = nyt_target / "data" / "nyt"
if nested_data.exists():
    for item in nested_data.iterdir():
        dest = nyt_target / item.name
        if dest.exists():
            continue
        shutil.move(str(item), dest)
    shutil.rmtree(nyt_target / "data", ignore_errors=True)

ensure_snapshot(
    repo_id="bert-base-cased",
    repo_type="model",
    target_dir=root / "pretrained_models" / "bert-base-cased",
    allow_patterns=[
        "*.bin",
        "*.json",
        "*.txt",
        "*.model",
        "tokenizer*",
        "vocab.txt",
    ],
)
PY

cd "$ROOT_DIR/tplinker"
python train.py "$@"
