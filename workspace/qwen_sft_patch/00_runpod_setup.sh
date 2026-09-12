#!/usr/bin/env bash
set -euo pipefail
# Run on a Linux x86-64 RunPod GPU with Python 3.11/3.12 and a CUDA-12.8-capable driver.
patch_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export HF_HOME=/workspace/hf-cache
export PIP_CACHE_DIR=/workspace/pip-cache
# Reuse the persisted environment on later Pods; never replace it silently.
if [ ! -x /workspace/sft-venv/bin/python ]; then
  sft_python="$(command -v python3.11 || command -v python3.12 || command -v python3)"
  "$sft_python" -c 'import sys; assert sys.version_info[:2] in {(3,11),(3,12)}, "Python 3.11 or 3.12 required"'
  "$sft_python" -m venv /workspace/sft-venv
fi
/workspace/sft-venv/bin/python -c 'import sys; assert sys.version_info[:2] in {(3,11),(3,12)}' 
/workspace/sft-venv/bin/python -m pip install 'pip==25.2'
/workspace/sft-venv/bin/python -m pip install 'torch==2.8.0' 'torchvision==0.23.0' --index-url https://download.pytorch.org/whl/cu128
/workspace/sft-venv/bin/python -m pip install -r "$patch_dir/requirements.txt"
/workspace/sft-venv/bin/python -m pip check
mkdir -p /workspace/sft_artifacts
/workspace/sft-venv/bin/python -m pip freeze > /workspace/sft_artifacts/environment.lock.txt
echo 'Setup complete. Run: source /workspace/sft-venv/bin/activate'
