# Minimal 7B bf16 LoRA patch — read this first

This ZIP is an overlay for your EXISTING `qwen_sft_patch` folder, not a standalone project. Keep all original files. It changes only the files listed below and supplies the required configs and a new continuation test. Your uploaded exporter, preparer, checkpoint verifier and telemetry code are byte-for-byte unchanged.

## Accepted settings

Qwen/Qwen2.5-VL-7B-Instruct; frozen BF16 base weights; LoRA rank 64/alpha 128 on language attention and MLP projections; frozen vision encoder and merger; two epochs for pilot and each main stage; three arms with identical configurations.

To minimize changes, accumulation stays 8, learning rate 1e-4, dropout 0, weight decay .01, warmup .10, SDPA attention, no packing, and the image-token budget stays 256–1280 with an 8192 expanded sequence cap. No FlashAttention/Liger dependency is added. These limits must pass coverage and GPU smoke checks; 80 GB does not guarantee every sequence fits. Do not interpret the unchanged image budget as preserving every detail of tall screenshots.

The base weights are BF16; PEFT may retain trainable adapter weights in FP32 for stability. This is bf16 LoRA, not four-bit training. Prompt and image tokens remain masked; assistant HTML and terminators are supervised.

## 1. On your Windows PC: replace the specific files

1. Download `qwen_sft_7b_minimal_patch.zip`.
2. Open PowerShell in your project root — the folder containing `qwen_sft_patch`, not inside `src/pipeline`.
3. Replace the ZIP path below if your browser saved it elsewhere:

```powershell
Expand-Archive -Path "$env:USERPROFILE\Downloads\qwen_sft_7b_minimal_patch.zip" -DestinationPath . -Force
```

The ZIP contains a `qwen_sft_patch` folder, so extraction replaces matching files and retains the rest. Do not put a second `qwen_sft_patch` folder inside the first. The configs must be in `qwen_sft_patch/configs/`.

Changed executable files: `00_runpod_setup.sh`, `01_check_environment.py`, `02_download_model.py`, `14_train_sft.py`, `15_report_sft_compute.py`, `run_all_arms.py`, `sft_core.py`, `requirements.txt`.

Updated support files: `README.md`, `TESTING.md`, `SHA256SUMS.txt`, `configs/pilot_100.json`, `configs/scale_10k.json`. Added: `configs/stage1_2000.json`, `configs/stage2_7500.json`, `tests/test_continuation.py`, and this note. The scale_10k file is retained for compatibility; use the explicitly named stage configs for your plan.

## 2. Before deploying

Use one A100 SXM 80 GB, the selected RunPod PyTorch 2.8.0 template, 20 GB container disk, and your ACTUAL 100 GB network volume. The latest screenshot selected ordinary Volume disk: switch to Network volume and choose the named volume if retaining data across Pod termination is still your plan. Enable SSH, with your public key registered and direct TCP SSH available. Jupyter is optional.

Only deploy when your bundle and this patch are ready, to avoid idle GPU charges. Keep all reusable files under `/workspace`. Do not delete a volume containing model weights or checkpoints.

## 3. Transfer using your existing SSH connection

In PC PowerShell, from the project root, after the `runpod-sft` SSH alias points at the current Pod:

```powershell
ssh runpod-sft "mkdir -p /workspace/qwen_sft_patch"
scp -r .\qwen_sft_patch\* runpod-sft:/workspace/qwen_sft_patch/
ssh runpod-sft
```

You are now typing on the GPU server. If you have not configured that alias yet, use the actual host and external port from RunPod's SSH-over-exposed-TCP command to set it up as in your SSH guide. Recreated Pods may have a new host/port. Never upload your private SSH key.

## 4. On the GPU: install and check, then reuse the saved model

```bash
nvidia-smi --query-gpu=name,memory.total --format=csv
cd /workspace/qwen_sft_patch
bash 00_runpod_setup.sh
source /workspace/sft-venv/bin/activate
export HF_HOME=/workspace/hf-cache
export CUDA_VISIBLE_DEVICES=0
python 01_check_environment.py --out /workspace/sft_artifacts/environment_7b.json
python 02_download_model.py --out /workspace/sft_artifacts/qwen_model_lock.json
```

Setup supports Python 3.11 or 3.12; it does not install a missing interpreter. A missing `venv` module needs the matching OS package. Reuse the same template and persisted venv on later Pods. Do not reinstall or upgrade packages between arms. The environment checker exercises real BF16 LoRA gradients but is not the full-model test.

The downloader verifies and reuses an existing 7B lock without downloading again. If this path contains a previous 3B lock, it intentionally refuses: use a new path, e.g. `/workspace/sft_artifacts/qwen7b_model_lock.json`, and use that same new path in ALL subsequent commands. Keep the snapshot and its cache blobs together on the volume. A new model download is necessary once when moving from 3B to 7B.

## 5. On the GPU: prepare the 100 pages, smoke-test, then run the pilot

Your existing exporter is unchanged. Export/upload/extract a 100-page training bundle using the README, so `/workspace/sft_bundle_100/bundle.json` exists. The 100 IDs must come from the initial 2,000 training IDs, not the 500 test IDs. Historical overwritten batch manifests must be restored/consolidated before export; the exporter cannot recover missing provenance.

```bash
python 13_prepare_sft.py --bundle /workspace/sft_bundle_100 --config configs/pilot_100.json --model-lock /workspace/sft_artifacts/qwen_model_lock.json --out /workspace/prepared_100_7b
python run_all_arms.py --prepared /workspace/prepared_100_7b --config configs/pilot_100.json --model-lock /workspace/sft_artifacts/qwen_model_lock.json --out /workspace/sft_artifacts/smoke_100_7b --smoke
```

If preparation reports overflow, inspect `coverage.csv` and `preparation_issues.json`. Do not truncate. If you change image/sequence policy, update all stage configs consistently, re-prepare into a new directory, and repeat the checks. The original 100-page statistics do not establish the sequence distribution of the full 2,000-page pool.

Only after all three smoke arms complete:

```bash
python run_all_arms.py --prepared /workspace/prepared_100_7b --config configs/pilot_100.json --model-lock /workspace/sft_artifacts/qwen_model_lock.json --out /workspace/sft_artifacts/pilot_100_seed42
python 16_verify_sft_checkpoint.py --run /workspace/sft_artifacts/pilot_100_seed42/original
python 16_verify_sft_checkpoint.py --run /workspace/sft_artifacts/pilot_100_seed42/naive
python 16_verify_sft_checkpoint.py --run /workspace/sft_artifacts/pilot_100_seed42/verified
python 15_report_sft_compute.py --runs /workspace/sft_artifacts/pilot_100_seed42 --out /workspace/sft_artifacts/report_pilot_7b
```

Use a `tmux` session for long commands if you want to disconnect SSH; closing SSH is not a billing stop. The pilot has 26 updates and 200 presentations per arm. Review measured cost and memory before the larger run. These commands refuse to overwrite run directories; use new output names for retries.

## 6. The two main stages

Follow README section 12 for complete commands: stage 1 is 2,000 training pages from the initial 2,500; the remaining 500 are held out. Stage 1 starts from the base, not the pilot adapters. Stage 2 uses 7,500 disjoint new pages and `--init-from-runs /workspace/sft_artifacts/stage1_seed42`. Each arm continues its own adapter with a fresh optimizer and two-epoch schedule. `--resume` is only for interruption recovery on the same dataset and recipe.

The continuation check requires the same base revision, seed, LoRA settings, image/sequence policy and optimizer recipe. It verifies saved artifact hashes and rejects reused page IDs. It does not implement site/domain splitting or detect duplicated content under new IDs. Keep original stage run folders, prepared datasets, model lock/cache and code unchanged for resumption. For later seeds, change the seed consistently in all configs and use separate run folders and same-seed predecessors.

## Validation limits

15 CPU tests passed, including valid continuation, wrong-arm rejection, reused-page rejection, wrong revision/rank rejection and corrupted-adapter rejection. Python 3.11 syntax and shell syntax passed. This environment did not execute model download, real processor expansion, CUDA training or real PEFT checkpoint continuation. Those remain RunPod checks; no GPU memory/runtime result is claimed.
