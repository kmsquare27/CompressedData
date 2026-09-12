# Steps 17 and 18 — additive compute extensions

## Install

Extract `qwen_compute_steps_17_18.zip` into the folder **containing** your existing
`qwen_sft_patch` directory. The result should be:

```
qwen_sft_patch/
  sft_core.py
  14_train_sft.py
  15_report_sft_compute.py
  compute_tools/
    17_compression_cost.py
    18_inference_profile.py
    README_STEPS_17_18.md
    tests/test_compute_extensions.py
```

Keep the `compute_tools` subfolder. Do not move these scripts into the parent:
the current trainer hashes top-level Python files for paired comparisons.
This addition changes none of your existing scripts, configurations or Step 11.
It targets the saved **Qwen2.5-VL-7B bf16 LoRA** patch and its `sft_core.py` API.

Step 17 uses Python's standard library only. Step 18 uses the existing pinned
RunPod training environment; no additional package installation is needed.

Run CPU checks from inside `qwen_sft_patch`:

```powershell
python -m unittest discover -s compute_tools/tests -v
```

## Step 17: recording compression costs

This has two modes: `record` executes a real command and records its elapsed
time; `report` calculates break-even using the saved ledger and Step 15 results.
It cannot reconstruct past timing and does not execute a made-up compression
pipeline. Command success does not prove which pages it processed: ensure its
input manifest actually matches the declared page-ID file.

### A. Before your measured compression run

Create a text file containing the exact input page IDs, one ID per line.
For example, from the CSV manifest you intend to use (replace its path):

```powershell
Import-Csv "D:\Thesis\YOUR_INPUT_MANIFEST.csv" | Select-Object -ExpandProperty page_id | Set-Content -Encoding ascii "D:\Thesis\measured_page_ids.txt"
```

Use a fresh run ID for each complete timing repetition. Wrap each real
production stage, including final validation and candidate attempts that fail
the visual gate. Do not benchmark a resume/cache-skip path as full compression.
Use a separate output directory or a safe isolated project copy for a rerun;
this wrapper does not change the underlying pipeline's overwrite behavior.

Example **Windows PowerShell**, one line (replace paths and the pipeline command
with the command you actually use):

```powershell
python "D:\Thesis\qwen_sft_patch\compute_tools\17_compression_cost.py" record --ledger "D:\Thesis\compression_cost.jsonl" --run-id batch2500_r1 --variant verified --stage l1 --page-ids "D:\Thesis\measured_page_ids.txt" --hardware-label "my Windows PC" --cwd "D:\Thesis\YOUR_PROJECT_ROOT" -- python src/pipeline/02_run_level1.py --source webcode2m
```

Repeat for the actual L2/L3/selection/final-validation commands, changing
`--stage` to unique labels (`l2`, `l3`, `select`, `validate`, etc.). The wrapper
records one stage duration and propagates the child command's exit code. It
does not invent a per-page measurement by dividing total time by page count.
Don't use `record` to pass `&&`, pipes, or PowerShell functions: it accepts an
executable followed by arguments, without a shell.

Measure the actual original-dataset preparation baseline with
`--variant original --stage baseline` (or multiple declared baseline stages).
Common screenshot acquisition can be excluded from both methods, or measured
in both, but use the same scope. Don't charge the verified arm for work also
required by the original arm. Separate stress testing/calibration and exploratory
census costs from production preparation, and state what you include.

The ledger records failures; it never silently overwrites old records. For the
canonical report, each declared stage must have exactly one successful record.
Failed attempts require a fresh run ID for a clean comparison. Preserve the
old records separately when discussing real operational costs.

### B. Optional real per-page timings

Per-page timing requires a small hook inside the existing page loop. The actual
compression source was not supplied for this addition, so it has not been edited.
Import the `StageTimer` class from Step 17 with `importlib`, then wrap the entire
work for each page:

```python
import importlib.util
spec = importlib.util.spec_from_file_location(
    "compression_cost", r"D:\Thesis\qwen_sft_patch\compute_tools\17_compression_cost.py")
cost = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cost)

# Inside your existing page loop:
with cost.StageTimer("page_cost.jsonl", run_id="batch2500_r1", stage="l1", page_id=page_id):
    process_page(page_id)  # Replace this with the EXISTING page work.
```

This is an integration example, not a function supplied by your pipeline.
For parallel workers, use a different page ledger for each worker. Do not add
overlapping page durations to sequential-stage elapsed time: that would double
count. Page detail is diagnostic; the reporter uses only `kind="stage"` records.

### C. After measured training and Step 15

Run Step 15 as usual. Step 17 requires its `compute_summary.csv`, including the
reconciliation/eligibility checks. Run the report on the same machine as Step 15
because that CSV identifies sessions by their paths. Copy the compression ledger
to RunPod if training and Step 15 were run there.

Example **RunPod**, from `/workspace/qwen_sft_patch` (replace run/session paths
and the declared stage lists with your actual ones):

```bash
python compute_tools/17_compression_cost.py report \
  --ledger /workspace/compression_cost.jsonl \
  --run-id batch2500_r1 \
  --original-stages baseline \
  --verified-stages l1 l2 l3 select validate \
  --original-summary /workspace/runs/ORIGINAL_SESSION/summary.json \
  --verified-summary /workspace/runs/VERIFIED_SESSION/summary.json \
  --compute-summary /workspace/compute_report/compute_summary.csv \
  --scope-note "Production preparation for 2500 pages; training on 2000 pages; common acquisition and research calibration excluded" \
  --out /workspace/break_even_r1.json
```

The output reports the cohort sizes explicitly; it does not pretend all prepared
pages were used for training. Baseline and verified preprocessing must cover
the same declared cohort; the training pair must also match.

For money estimates, supply the actual `--hourly-rate` and `--currency` on **each**
record command, and `--gpu-hourly-rate` with the same currency on `report`.
Rates are user-supplied assumptions, not fetched prices. Missing prices produce
no monetary break-even result. A local computer isn't automatically free; use
elapsed-time analysis if you don't have a defensible monetary valuation.

Definitions:

- Incremental preparation = verified preparation minus original preparation.
- Seconds saved per epoch = difference in training-loop time excluding checkpoint
  writes, divided by the common measured epoch count.
- Elapsed break-even = incremental preparation seconds / saved seconds per epoch.
- Monetary break-even uses preparation costs and GPU cost savings in the same
  currency. Nonpositive training savings have no finite break-even estimate.
- This is a linear projection, not a measured guarantee over future epochs.
  It excludes model-load and checkpoint overhead from the marginal epoch estimate;
  training-job savings are also reported separately.
- Stage durations measure elapsed service time, not CPU core-seconds. Gaps
  between stage commands are excluded; wrap the whole production driver as one
  stage if you need its complete elapsed time (do not also sum its children).

## Step 18: generation and inference profiling

Run on the GPU after a completed non-smoke training run. Use a **held-out Step 12
bundle**, extracted as a directory containing `bundle.json`, `pages.jsonl` and its
page files. Do not use the training bundle. It reads screenshot/HTML pairs for
validation and references, but sends only the screenshot and fixed training
prompt to the model. It does not filter evaluation pages based on target length.

All bundle files, model files and adapter files are hash-verified. Current and
ancestor training page IDs must be disjoint from the held-out bundle. This is an
ID check: establish site/template/near-duplicate separation in the dataset split
yourself. Saved reference HTML is never inserted in the inference prompt.

Use the same held-out bundle, output budget, warmup settings, repeat count and
order seed for every arm. Choose the output budget using development/pilot data,
then freeze it before final evaluation. The default is 4096 generated tokens;
this is a starting configuration, not proof that every page fits. Context
overflow fails rather than silently truncating input or reducing the budget.

### A. One-page GPU check

From `/workspace/qwen_sft_patch`, in your activated training environment:

```bash
python compute_tools/18_inference_profile.py \
  --run /workspace/runs/ORIGINAL_SESSION \
  --bundle /workspace/heldout_bundle \
  --model-lock /workspace/sft_artifacts/qwen7b_model_lock.json \
  --limit 1 --max-new-tokens 128 --warmup-requests 1 \
  --out /workspace/inference_check_original
```

`--run` must point to the **session directory** containing `summary.json`,
`recipe.json`, `final_adapter/`, and `final_adapter_hashes.json`, not the parent
containing several sessions. Use your actual model-lock path. The 128-token
check is deliberately short and cannot establish reconstruction quality.

### B. Full evaluation, one arm per process

```bash
python compute_tools/18_inference_profile.py \
  --run /workspace/runs/ORIGINAL_SESSION \
  --bundle /workspace/heldout_bundle \
  --model-lock /workspace/sft_artifacts/qwen7b_model_lock.json \
  --max-new-tokens 4096 --warmup-requests 2 --warmup-tokens 32 \
  --order-seed 42 --repeats 1 \
  --out /workspace/inference_original
```

Repeat with `NAIVE_SESSION` / `inference_naive`, then `VERIFIED_SESSION` /
`inference_verified`. Arm identity comes from the saved recipe; you cannot
mislabel an adapter with an `--arm` flag. Each process loads one base and one
adapter locally, so it never needs the blocked Hugging Face endpoint. Run one
arm at a time. Rotate arm order across separate experiment repetitions.

For cost estimates optionally add `--gpu-hourly-rate YOUR_ACTUAL_RATE
--currency USD`. These estimates use measured time and an explicit rate; they
exclude idle rental time, storage and transfer charges.

### C. Compare completed profiles (CPU only)

```bash
python compute_tools/18_inference_profile.py \
  --compare /workspace/inference_original /workspace/inference_naive /workspace/inference_verified \
  --out /workspace/inference_comparison
```

This produces paired per-page/repeat metrics plus aggregate savings. It rejects
different model revisions, training configurations/cohorts/code/predecessors,
evaluation inputs, generation settings, GPU models or software versions. It also
checks that every expected request exists and the request ledger is unchanged.
It doesn't control hardware clocks, other tenants or transient load: record and
manage those experiment conditions yourself.

### Outputs and timing definitions

| File / field | Meaning |
|---|---|
| `generated/*.raw.txt` | Decoded generation before fence cleanup |
| `generated/*.html` | Renderable candidate; only a single enclosing Markdown fence is removed |
| `generated/*.tokens.json` | Exact generated token IDs including terminal EOS |
| `requests.jsonl` / `.csv` | Per-page/repeat outputs, timings and reference paths |
| `profile.json` | Identity, settings, environment, aggregates and completion status |
| `ttft_generate_seconds` | Start of `generate()` to first generated token's host callback |
| `ttft_request_seconds` | Image loading/preprocessing start to that callback |
| `first_to_last_token_seconds` | Time between first and last token arrivals |
| `decode_tokens_per_second` | `(generated_tokens - 1) / first_to_last_token_seconds`; null for one token |
| `generation_seconds` | Synchronized `generate()` wall time |
| `request_seconds` | Image open/preprocessing, transfer, generation and final token decoding; excludes saving files |
| `peak_allocated_bytes` / `peak_reserved_bytes` | Absolute PyTorch process allocator peaks including the resident model; not NVML device occupancy |
| `hit_output_limit` | Output length reached the budget, even if EOS occurred exactly at it |
| `truncated_by_limit` | Budget exhausted without terminal EOS |

The token streamer avoids text-chunk TTFT ambiguity. Its host callbacks and
CPU transfers introduce overhead, so report **instrumented streaming latency**.
TTFT includes image encoder work and generation setup; it is **not** isolated
prefill kernel time. No separate isolated prefill metric is invented. Warmup
requests are excluded from per-request summaries; total job time includes them
and model loading, but excludes initial file verification and final report write.

The allocator cache is emptied once after warmup, never between measured pages;
reserved-memory peaks can depend on earlier requests. The same page order is
used in each arm. No peak-memory subtraction is presented as a physical GPU
saving. Timing repeats are not independent training seeds, and p50/p95 across
requests are distribution summaries, not confidence intervals over training jobs.

Failures stop the profile and preserve partial logs. A failed or incomplete run
cannot enter the paired report. Truncated outputs remain included and marked;
do not remove difficult pages to improve a headline number.

## Reconstruction quality is a separate required evaluation

Render the saved generated HTML using the same deterministic harness and join
quality scores by `page_id` (and repeat). This script supplies the generation
stage for RQ2, but does not implement Design2Code metrics or claim pixel fidelity.
Shorter HTML alone is not success. Report quality, empty outputs, failures and
truncation alongside latency/token savings. Comparison CSVs explicitly flag
truncated pairs, and `comparison.json` says `quality_evaluated: false`.

## Validation and limitations

CPU tests cover real subprocess failure recording, elapsed/cost arithmetic,
duplicate/missing/overlapping stage rejection, per-page exception propagation,
streamer prompt exclusion, token timing, EOS/truncation, output preservation,
held-out ID overlap/hash checks, and inference comparison ledger integrity.
Python syntax and CLI help are checked. Actual 7B GPU execution, timing overhead,
GPU memory and quality are **not validated in this development environment**;
run the short GPU check above before the full evaluation.

Reference for the pinned stream interface:
https://huggingface.co/docs/transformers/v4.57.1/en/internal/generation_utils#streamers
