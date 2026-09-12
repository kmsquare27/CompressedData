> Updated for the accepted 7B bf16 LoRA recipe. Read `PATCH_NOTES.md` first for exact PC/SSH replacement steps. Do not follow an older 3B guide. Unchanged defaults: accumulation 8, dropout 0, weight decay .01, warmup .10, SDPA, image budget 256–1280, sequence cap 8192. These were intentionally not retuned. Freeze any justified policy changes before main training.

# From finalized pages to a 100-page Qwen SFT pilot

This is an additive execution patch for your accepted thesis plan. It exports your finalized screenshot/HTML pairs, prepares three matched training arms, fine-tunes Qwen with bf16 LoRA, and records computational measurements. It stops after training and adapter verification. Test-set evaluation, generation, Design2Code scoring and mutation testing remain a later stage.

**Use RunPod for this pilot. Your PC's 8 GB RAM and 2 GB RTX GPU are not a practical configuration for this vision-language training recipe.** A 100-page dataset shortens training but does not substantially reduce the peak memory required by one long screenshot/HTML example.

The implemented model is **`Qwen/Qwen2.5-VL-7B-Instruct`**, supporting screenshot input. Your gate uses the 3B tokenizer: compare actual HTML token IDs/counts before treating historical counts as identical to this model’s counts. A text-only Qwen2 checkpoint cannot consume these screenshots. This patch does not silently substitute a different architecture.

The patch and CPU integrity tests have been checked in the authoring environment. The actual model download, processor behavior, CUDA training and real peak memory still require the included RunPod checks. No training results or hardware guarantees are fabricated.

## 1. The experiment you will run

| Item | Fixed pilot setting |
|---|---|
| Input | The same original harness screenshot for every arm |
| Training examples | 100 distinct finalized page IDs, all used for training |
| Original arm | Original HTML bytes |
| Naive arm | Generic `minify-html` result, without your gate |
| Verified arm | Frozen Step 08 target; includes original fallbacks where necessary |
| Starting weights | The same immutable Qwen model commit; a fresh adapter for each arm |
| Trainable weights | Language-decoder LoRA adapters only; vision encoder and base weights frozen |
| Quantization | None; frozen base weights loaded in BF16 |
| Epochs / seed | 2 / 42 |
| Microbatch / accumulation | 1 / 8; no sequence packing |
| Updates | 26 per arm: 13 per epoch, with a final four-example window |
| Loss | Assistant target only; example-normalized within each update |
| Target includes | HTML and assistant end-of-turn tokens |
| Sequence limit | 8,192 **expanded** positions including image and prompt |
| Image policy | Processor min/max pixels equivalent to 256–1,280 merged image tokens |
| Optimizer | AdamW; learning rate 1e-4, weight decay .01, cosine schedule, 10% warmup |
| LoRA | Rank 64, alpha 128, dropout 0 |
| Checkpoint choice | Fixed final epoch; no own-arm validation-loss selection |

These are pilot starting settings, not proven optimal hyperparameters. The pilot establishes that the data, learning path, saving and measurement pipeline work. It does not establish model quality or a publishable advantage from 100 pages.

The naive policy enables CSS minification, disables JavaScript minification, retains closing tags and HTML/head opening tags, and disables doctype shortening. It is a declared generic minifier baseline, not a deliberately maximally destructive baseline. A minifier exception uses original HTML and records the exception; a poor visual result is retained. The verified arm is the final selected representation from your pipeline, not necessarily the most aggressive level.

## 2. Files and where they run

| File | Run it on | Purpose |
|---|---|---|
| `12_export_sft_bundle.py` | Your source PC | Validate Step 08 artifacts and make a portable paired bundle |
| `00_runpod_setup.sh` | RunPod | Create Python environment and install pinned main dependencies |
| `01_check_environment.py` | RunPod | Confirm CUDA, BF16 and a real tiny LoRA backward pass |
| `02_download_model.py` | RunPod | Download once and freeze model commit and file hashes |
| `13_prepare_sft.py` | RunPod | Create naive targets, count actual processor tokens and check lengths |
| `14_train_sft.py` | RunPod | Train one arm and record timing, memory, tokens and checkpoints |
| `run_all_arms.py` | RunPod | Run all three arms sequentially |
| `15_report_sft_compute.py` | RunPod | Export measured cost tables and PNG/PDF figures |
| `16_verify_sft_checkpoint.py` | RunPod | Verify saved adapters without inference |
| `configs/pilot_100.json` | RunPod | Pilot settings |
| `configs/scale_10k.json` | RunPod | Same main recipe; longer checkpoint interval |
| `sft_core.py`, `telemetry.py` | Imported helpers | Data contracts, masking, accounting and telemetry |

Keep the entire `qwen_sft_patch` folder together. Do not paste these files into your existing `src` tree or replace your accepted file 11. Your uploaded markdown describes a merged file 11 that was not itself supplied, so this patch leaves that work intact.

## 3. Prepare the 100 pages on your PC

Extract `qwen_sft_patch.zip` into your compression project root. You should have `<your-project>/qwen_sft_patch/README.md`.

Before exporting, your existing pipeline must have produced:

```text
reports/csv/final_validation_webcode2m.csv
reports/csv/final_validation_webcode2m_environment.json
data/splits/compressed_webcode2m_manifest.csv
reports/csv/level_selection_webcode2m.csv
```

The pilot manifest can supply original HTML paths if the selection report is unavailable. The frozen final HTML and original harness PNG files referenced by Step 08 must still exist. The exporter checks recorded HTML hashes, cross-file IDs/counts/levels/paths, `status=ok`, and `final_pixel_identical=True`. A recorded failed repeatability check is rejected. Original fallbacks are valid zero-compression examples.

If Step 08 has not run, use your project's existing command from its existing environment:

```powershell
python src/pipeline/08_final_validate.py --source webcode2m --repeat
```

Do not add `--tolerance` for this strict pilot. In the uploaded implementation, rerunning Step 08 recreates its final target/render directories. Export the immutable bundle before any later rerun changes those outputs.

Run the export from the project root in PowerShell. This exporter uses the standard library only, so it does not need PyTorch or your GPU. Python 3.10+ is needed on the PC; RunPod uses 3.11.

```powershell
python qwen_sft_patch/12_export_sft_bundle.py --root . --source webcode2m --n 100 --seed 42 --out sft_bundle_100 --zip
```

The result is `sft_bundle_100/` and `sft_bundle_100.zip`. Original/verified HTML are copied byte-for-byte; images are identical across arms. CSV provenance is included. The image hash is first recorded during export because the uploaded Step 08 did not record a screenshot hash; this does not retrospectively authenticate the screenshot at Step 08 time.

If you already have exact training IDs, put one ID per line in `train_100_ids.txt` and append `--ids train_100_ids.txt`. If reserved validation/test IDs already exist, append `--exclude-ids reserved_ids.txt`. The script rejects overlap. All 100 pages here are training pages; there is no hidden 80/20 split and no test-set use. Reserve future test sites/templates before scaling, and do not later call these training/pilot pages a held-out test set.

If fewer than 100 pages pass finalization, the exporter stops. Finalize more pages or explicitly choose a smaller pilot with `--n`; do not report a smaller successful export as 100 pages.

## 4. Choose and connect to RunPod

Use **one A100 SXM 80 GB** for the pilot and sequential comparison arms. This is the selected target hardware, not a guarantee that every sequence fits. Run the full-model smoke checks before committing to full training.

Use Linux x86-64, a BF16-capable NVIDIA GPU, a current driver compatible with CUDA 12.8, and a RunPod PyTorch template providing Python 3.11 or 3.12. The setup creates its own virtual environment instead of relying on a template's preinstalled Python packages. Select the template by these properties; template names and availability can change.

Provision roughly **100 GB persistent workspace storage initially**. Model weights, environment, paired images, prepared copies and checkpoints consume disk independently of GPU memory. Check actual usage before 10k; there is no guarantee 100 GB fits that dataset.

A RunPod network volume attaches at Pod creation and persists independently of the Pod. The usual mount is `/workspace`. A Pod volume disk and a network volume have different deletion behavior; verify your chosen storage before relying on it. See the official storage reference below. No deployment or billing action is performed by this patch.

Open RunPod's Connect panel and use its JupyterLab or SSH connection. Upload these two archives to `/workspace` using JupyterLab's file browser:

```text
qwen_sft_patch.zip
sft_bundle_100.zip
```

In a RunPod terminal:

```bash
cd /workspace
python3.11 -m zipfile -e qwen_sft_patch.zip /workspace
python3.11 -m zipfile -e sft_bundle_100.zip /workspace
cd /workspace/qwen_sft_patch
nvidia-smi
python3.11 --version
bash 00_runpod_setup.sh
source /workspace/sft-venv/bin/activate
export HF_HOME=/workspace/hf-cache
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=0
python 01_check_environment.py --out /workspace/sft_artifacts/environment_check.json
python -m unittest discover -s tests -v
```

Keep these environment variables in every new terminal used for this experiment. `CUDA_VISIBLE_DEVICES=0` assumes your assigned GPU is index 0 in the Pod. The script expects exactly one visible GPU.

The setup accepts Python 3.11 or 3.12 and reuses `/workspace/sft-venv` on later Pods. Keep the same Python version/container image across arms and stages. If venv support is missing, install the OS venv package matching that interpreter (for Ubuntu 24.04 Python 3.12: `apt-get update` then `apt-get install -y python3.12-venv`). Run the environment check again after reconnecting to a new Pod.

For a persistent terminal session, use `tmux new -s sft100` if tmux is available, then re-enter the activation/export commands above. Detach with Ctrl-B then D; reattach with `tmux attach -t sft100`. This helps with browser/SSH disconnects; it does not prevent a Pod from stopping.

## 5. Download and freeze the base model once

```bash
cd /workspace/qwen_sft_patch
python 02_download_model.py --out /workspace/sft_artifacts/qwen_model_lock.json
```

This resolves the model revision to a commit SHA, downloads the complete snapshot, and hashes its files. All arms then load locally from that same snapshot. Keep `qwen_model_lock.json` and the model cache. A new machine needs the same commit downloaded there; regenerate the lock with `--revision <recorded-commit>` and a new output path if the absolute cache path changes. Preserve the recorded SHA, rather than resolving `main` again for each arm.

Installation and model download are one-time setup costs. They are excluded from per-arm training-job latency and are not presented as compression savings.

## 6. Prepare all three arms and inspect the token report

```bash
python 13_prepare_sft.py --bundle /workspace/sft_bundle_100 --config configs/pilot_100.json --model-lock /workspace/sft_artifacts/qwen_model_lock.json --out /workspace/prepared_100
```

This uses the actual downloaded Qwen processor and tokenizer. It produces:

```text
prepared_100/prepared.json
prepared_100/original.jsonl
prepared_100/naive.jsonl
prepared_100/verified.jsonl
prepared_100/coverage.csv
prepared_100/token_summary.csv
prepared_100/preparation_issues.json
prepared_100/pages/...
prepared_100/processor/...
```

Open `token_summary.csv` and `coverage.csv` in JupyterLab. Confirm 100 pages per arm, identical image token counts for each corresponding page, no over-limit sequences, and acceptable naive exception counts. The preparation script checks the exact prompt-token prefix and masks it; it never locates a target by searching your HTML for an assistant-role string. If tokenization merges the template newline with leading HTML whitespace, exact full-sequence offsets are verified and that entire whitespace token is masked. The number of affected target whitespace characters is recorded as `masked_boundary_whitespace_chars`; non-whitespace boundary merges fail. Reserved model token strings in HTML are rejected explicitly.

The prompt asks for screenshot-to-HTML generation, and the response contains raw HTML, without Markdown fences. It does not impose a new self-contained-assets requirement on your existing targets. Images are your **harness renders**, which may include the harness's asset substitutions; state this task definition in the thesis rather than describing them as untouched WebCode2M screenshots.

Image downsampling can make very tall pages hard to read. Inspect representative long screenshots and the processor's recorded grids. Changing the image-token cap changes the learning problem: freeze it across arms and acknowledge it when interpreting later quality results.

**If sequences exceed 8,192:** preparation writes coverage/issues and deliberately stops without a READY dataset. Do not truncate HTML. To retain the exact 100 pages, increase `max_seq_length` in a copied configuration, use a GPU that can support it, and re-prepare into a new directory. The original arm must fit too. More accumulation does not make a single overlong example fit.

An alternative is a predeclared common-length eligibility rule, selecting a different set of 100 from a larger finalized pool. This changes the population and must be disclosed. `--overflow-policy common-fit` removes every over-limit page from **all** arms and records exclusions, but it may leave fewer than 100. It does not silently refill the sample or still claim 100. Keep default `error` for the first attempt.

Preparation has CPU cost and is timed separately. It processes one image at a time; it does not hold every page's pixel tensors in RAM. The portable bundle and prepared directory contain duplicate copies for immutability, so budget disk accordingly.

## 7. Run the GPU smoke tests before the full pilot

```bash
python run_all_arms.py --prepared /workspace/prepared_100 --config configs/pilot_100.json --model-lock /workspace/sft_artifacts/qwen_model_lock.json --out /workspace/sft_artifacts/smoke_100 --smoke
python 16_verify_sft_checkpoint.py --run /workspace/sft_artifacts/smoke_100/original
python 16_verify_sft_checkpoint.py --run /workspace/sft_artifacts/smoke_100/naive
python 16_verify_sft_checkpoint.py --run /workspace/sft_artifacts/smoke_100/verified
```

Each arm runs two optimizer updates, covering examples selected for longest sequence, most vision patches and largest sequence-times-patch size. It exercises image processing, real loss/backward, accumulation, AdamW and adapter saving. It checks finite gradients/loss and that adapter weights changed. The verifier checks saved file hashes, finite weights and nonzero learned LoRA B matrices. The base is reloaded independently for each arm.

This is a practical memory screen, not a proof that every full run will fit. Full runs retain explicit failure logs. Smoke directories are marked diagnostic and automatically excluded from paired research savings. The full pilot starts fresh from the base; it does not continue the smoke adapter.

**If you see CUDA out of memory:** inspect `summary.json`, the latest printed peak, and `nvidia-smi`. Ensure no other training process occupies the GPU. Prefer moving to a larger GPU while keeping the data policy unchanged. Lowering image resolution is a possible new experimental recipe, but requires re-preparing and rerunning all three arms. Microbatch is already one; increasing accumulation only changes the effective update size. The script does not secretly offload, shorten targets, or reduce image resolution to get past OOM.

## 8. Run the full 100-page SFT pilot

```bash
python run_all_arms.py --prepared /workspace/prepared_100 --config configs/pilot_100.json --model-lock /workspace/sft_artifacts/qwen_model_lock.json --out /workspace/sft_artifacts/pilot_100_seed42
```

The launcher runs original, naive and verified sequentially on the same GPU. Expect 26 updates and 200 example presentations **per arm**, giving 78 updates and 600 presentations across the three-arm experiment. Duration depends on the actual sequence distribution and GPU; no credible number of minutes can be promised before measurement.

If you first want only your verified dataset's full training run, use:

```bash
python 14_train_sft.py --prepared /workspace/prepared_100 --config configs/pilot_100.json --model-lock /workspace/sft_artifacts/qwen_model_lock.json --arm verified --out /workspace/sft_artifacts/verified_only_100
```

That produces a fine-tuned adapter, but a compression-related computational comparison requires the matched original/naive runs. Do not run both this optional command and the complete three-arm command unintentionally; they are separate billed training runs.

Training labels supervise only the assistant suffix. For microbatch losses `L_i`, the main update loss is `sum(L_i)/number_of_examples_in_this_window`. Thus, the last four-example window divides by four, not eight. To run the accepted loss-normalization ablation later, set `loss_normalization` to `token` in a copied config: the weight becomes `target_tokens_i / sum(target_tokens_in_window)`. Re-run all arms under that objective and keep it a separate experiment. Neither objective removes all optimization differences caused by different target representations.

Your own-arm training losses are optimization diagnostics. Their token distributions differ, so a lower verified-arm loss is not evidence that it generates better pages. All arms save the fixed final epoch; there is no best-checkpoint decision based on those values.

## 9. Verify the trained artifacts

```bash
python 16_verify_sft_checkpoint.py --run /workspace/sft_artifacts/pilot_100_seed42/original
python 16_verify_sft_checkpoint.py --run /workspace/sft_artifacts/pilot_100_seed42/naive
python 16_verify_sft_checkpoint.py --run /workspace/sft_artifacts/pilot_100_seed42/verified
```

Each completed arm directory contains:

```text
summary.json                  completion status, timings, memory, counters, environment
recipe.json                   data/model/config/code identity
steps.jsonl                   completed optimizer updates and page order
microbatches.jsonl            successfully completed backward events
telemetry.jsonl               sampled power, utilization, device memory, process RSS
trainable_parameters.json     exact modules and parameter count
checkpoints/step_.../          adapters plus optimizer/scheduler/RNG/progress
latest_checkpoint.json        most recent completed checkpoint path
final_adapter/                trained LoRA adapter, processor, training recipe
final_adapter_hashes.json     artifact integrity hashes
```

`final_adapter` is the fine-tuning result, **not a standalone copy of the entire base model**. Keep the immutable base-model revision and processor with it. Saving/merging into a full model and running inference are intentionally deferred.

Download/back up the complete per-arm run directories, model lock, environment check/freeze, prepared metadata/manifests and exported bundle before removing your Pod or storage. The full model cache can be re-downloaded by commit if necessary. The run data and adapters are the evidence you must preserve.

## 10. Generate the computational thesis outputs

```bash
python 15_report_sft_compute.py --runs /workspace/sft_artifacts/pilot_100_seed42 --out /workspace/sft_artifacts/compute_report_100
```

It writes `compute_summary.csv`, `paired_savings.csv`, `report_issues.json`, `METRIC_DEFINITIONS.md` and matched-arm training-cost figures in PNG and PDF. Comparisons require exactly one successful fresh run of each arm with matching data, model commit, config, code and relevant environment. Failed, smoke and resume sessions remain reportable as raw sessions but do not enter the primary savings calculation. Do not modify patch code halfway through the three-arm run.

| Research measure | Recorded meaning |
|---|---|
| End-to-end training-job latency | Imports/preflight, model load, training, logging, checkpoint IO and final save; setup/download/shared preparation excluded |
| Training-loop time | Both with and without checkpoint IO |
| Update latency | Synchronized wall time per optimizer update; p50/p95 after configured warmup |
| Peak GPU memory | PyTorch allocated/reserved peaks; separate sampled NVML total device memory |
| Actual tokens processed | Expanded image + prompt + supervised assistant positions, repeated across epochs |
| Supervised tokens | Non-masked target positions after causal shift |
| Vision workload | Pre-merge image patch count, reported separately from decoder positions |
| Throughput | Sequence tokens/s, target tokens/s and examples/s; denominator stated |
| Host memory | Sampled process RSS |
| GPU utilization | Time-weighted available NVML samples |
| Energy | Sampled device power integration, with covered seconds and missing-sensor information |
| Failures | OOM/errors, partial successful backward events, completed updates and last checkpoint |

Never add vision patch count to decoder sequence positions as if they were one token unit. Never count checkpoint recomputation as new data exposure. No FLOPs, cloud-price estimates, inference latency, time-to-first-token or test quality are inferred from these logs.

For savings, the report uses `100 * (original_cost - arm_cost) / original_cost`; negative values are real regressions, not clipped to zero. Wall-time speedup is `original_seconds / arm_seconds`. Lower HTML token count need not imply a proportional speedup: screenshot processing, fixed model state, sequence shape and logging contribute to total cost. Tokens/s can even decrease while examples/s improves, so report both.

Use a dedicated GPU and keep checkpoint frequency, image policy, objective, sequence limit, precision and hardware fixed across arms. For thesis-level uncertainty, run at least three predeclared seeds on the same GPU type and report each paired difference plus the mean/spread. Change seed consistently across all arms, use a new run root, and vary arm order with `--arms verified original naive`, for example, to reduce systematic cache/thermal order effects. One pilot run provides engineering measurements, not confidence intervals over repeat training jobs.

For end-to-end project economics, separately measure compression/final-validation preprocessing and add it once at the chosen amortization scale. This patch records shared SFT preparation time but does not invent timing for your earlier compression pipeline. Do not charge the same shared preparation three times and call it arm-specific training cost.

## 11. Resume an interrupted job

Read the affected arm's `latest_checkpoint.json` and use its actual checkpoint path. Example:

```bash
python 14_train_sft.py --prepared /workspace/prepared_100 --config configs/pilot_100.json --model-lock /workspace/sft_artifacts/qwen_model_lock.json --arm verified --resume /workspace/sft_artifacts/pilot_100_seed42/verified/checkpoints/step_000020 --out /workspace/sft_artifacts/verified_resume_01
```

The example step number must exist; the default pilot also saves at epoch ends (13, 26). Resume is supported only at a completed optimizer boundary, with exactly the original recipe. It restores adapters, optimizer, scheduler, sampler position and RNG state. A checkpoint at the last update is already the trained state; if only final saving was interrupted, its adapter is still usable and should be preserved.

Load only your own trusted checkpoints: optimizer/RNG state uses PyTorch's Python serialization. A new output directory prevents overwriting earlier timing evidence. Work since the last checkpoint may be repeated; sum per-session observed costs for operational accounting, never sum cumulative logical counters. For the main matched latency table, use a fresh uninterrupted rerun. Restored RNG does not guarantee bitwise-identical GPU kernels across machines.

If a launcher stops after one arm fails, use the single-arm command to run the remaining arms into new directories, then pass the explicit directories to the reporter. Do not rerun already successful arms merely to reconstruct one launcher directory.

## 12. Main stages: 2,000 pages, then 7,500 new pages

Freeze 2,000 training IDs and 500 held-out test IDs from the initial 2,500 pages. The pilot's 100 IDs are a subset of those 2,000 TRAIN IDs. Restart stage 1 from the locked base (no pilot adapter). Never train on the 500 test pages. Split by site/template and inspect near duplicates before exporting; these scripts do not perform domain splitting. Use globally unique page IDs across batches; the continuation check rejects overlapping IDs but cannot discover duplicated content under new IDs.

Export each stage using the existing exporter with its appropriate `--n`, `--ids`, and `--exclude-ids`, then upload/extract into `/workspace/sft_bundle_2000` and, later, `/workspace/sft_bundle_7500`. Do not use the older 10,000-page instructions: the final population is 9,500 training pages and 500 test pages.

Run the following on the GPU, inside `/workspace/qwen_sft_patch`, with the venv activated. Preparation fails on overlength pages rather than truncating. Review coverage for the entire stage-1 pool before freezing the image/sequence policy; 100 pilot pages do not establish the maximum. Update the SAME policy in all configs if necessary, then re-prepare and repeat the pilot before stage 1.

```bash
python 13_prepare_sft.py --bundle /workspace/sft_bundle_2000 --config configs/stage1_2000.json --model-lock /workspace/sft_artifacts/qwen_model_lock.json --out /workspace/prepared_2000
python run_all_arms.py --prepared /workspace/prepared_2000 --config configs/stage1_2000.json --model-lock /workspace/sft_artifacts/qwen_model_lock.json --out /workspace/sft_artifacts/smoke_2000 --smoke
python run_all_arms.py --prepared /workspace/prepared_2000 --config configs/stage1_2000.json --model-lock /workspace/sft_artifacts/qwen_model_lock.json --out /workspace/sft_artifacts/stage1_seed42
python 15_report_sft_compute.py --runs /workspace/sft_artifacts/stage1_seed42 --out /workspace/sft_artifacts/report_stage1
```

Stage 1 has 4,000 presentations and 500 optimizer updates per arm. Verify all three final adapters with `16_verify_sft_checkpoint.py --run /workspace/sft_artifacts/stage1_seed42/ARM`, replacing ARM with original, naive, then verified. Preserve complete run folders, not only adapter weights: continuation validates their manifests and recipes.

When ready for the additional 7,500 pages, reuse the same downloaded model and environment:

```bash
python 13_prepare_sft.py --bundle /workspace/sft_bundle_7500 --config configs/stage2_7500.json --model-lock /workspace/sft_artifacts/qwen_model_lock.json --out /workspace/prepared_7500
python run_all_arms.py --prepared /workspace/prepared_7500 --config configs/stage2_7500.json --model-lock /workspace/sft_artifacts/qwen_model_lock.json --out /workspace/sft_artifacts/smoke_7500 --smoke
python run_all_arms.py --prepared /workspace/prepared_7500 --config configs/stage2_7500.json --model-lock /workspace/sft_artifacts/qwen_model_lock.json --out /workspace/sft_artifacts/stage2_seed42 --init-from-runs /workspace/sft_artifacts/stage1_seed42
python 15_report_sft_compute.py --runs /workspace/sft_artifacts/stage2_seed42 --out /workspace/sft_artifacts/report_stage2
```

The stage-2 smoke uses disposable base adapters to check new sequence sizes. Actual continuation loading is checked when stage 2 starts. Each arm loads its own stage-1 adapter, retaining learned weights, but starts a NEW optimizer, scheduler and two-epoch counter. It does not merge or stack an additional adapter. Stage 2 has 15,000 presentations and 1,876 updates per arm (938 per epoch). The full staged run has 19,000 presentations per arm. It is sequential training, not a shuffled two-epoch run over the combined 9,500 pages.

`--resume` is ONLY for an interrupted run with unchanged data/config/code; `--init-from-runs` is for the new dataset stage. To resume an interrupted continuation, use `14_train_sft.py --resume ...` with the stage-2 config/data, not both options. Parent comparison identifiers keep stages and different predecessors out of the same report group. Report each stage separately; for total project compute, add stage times/tokens and take the maximum of their memory peaks. Shared setup/download costs are not multiplied by the arm count. GPU timings are unmeasured until execution.

Keep the sequence/image policy fixed in stage 2. If new pages do not fit, do not truncate or change one arm: use the declared common-fit policy for every arm and report exclusions, or define a separate revised experiment. The configurations do not enforce the nominal page counts in their names; confirm READY reports exactly the intended count.

## 13. References and implementation choices

The main dependency versions are pinned in `requirements.txt`; the setup writes the complete resolved environment to `/workspace/sft_artifacts/environment.lock.txt`. Preserve that file because transitive packages are resolved at installation. Do not mix prepared data and training from different installed versions; the trainer checks recorded package versions.

- [Qwen2.5-VL model and image-resolution interface, Transformers 4.57.1](https://huggingface.co/docs/transformers/v4.57.1/en/model_doc/qwen2_5_vl)
- [Qwen2.5-VL-7B-Instruct model card](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct)
- [PEFT adapter loading/saving](https://huggingface.co/docs/peft/v0.17.0/package_reference/peft_model)
- [PyTorch official prior-version installation commands](https://pytorch.org/get-started/previous-versions/)
- [minify-html configuration](https://github.com/wilsonzlin/minify-html)
- [RunPod storage types](https://docs.runpod.io/pods/storage/types)
- [RunPod connection methods](https://docs.runpod.io/pods/connect-to-a-pod)

Read `TESTING.md` for the exact validation boundary of this delivered patch. The included GPU checks are part of the execution process, not a claim that training already ran in the authoring environment.
