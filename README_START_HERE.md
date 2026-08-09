# START HERE — Day-One Guide

You have three documents from our earlier work (the decision report, the
execution protocol, the paper-reading plan). **This file is the only one you
need open today.** It turns the protocol's Stages 0–3 into runnable commands;
the protocol remains the reference for depth, and Stages 4+ (Level 2 rebuild,
Level 3, dataset scale-up, fine-tuning) begin only after the checkpoint at the
bottom of this file.

The one rule that resolves all confusion about "where to start":
**nothing in this project means anything until (a) the renderer is proven
deterministic and (b) the acceptance gate exists and is stress-tested.**
Every level, table, and figure sits on those two things. So we build them
first. Levels come after.

---

## What is in this kit

| File | What it is | Protocol stage |
|---|---|---|
| `src/render/harness.py` | Deterministic renderer (fixed viewport, full-page, frozen animations, intercepted network) — fixes all four bugs of the old renderer | Stage 1 (Level 0) |
| `src/compare/gate.py` | Acceptance Gate v2 (G0 parse → G1 tokens → G2 height → G3 text → G4 blocks/IoU → G5 ΔE00 → G6 optional LPIPS; SSIM demoted to diagnostic) | Stage 2 |
| `config/gate_config.yaml` | Provisional thresholds — frozen only after the stress test | Stage 2 |
| `src/compress/level1_minify.py` | Upgraded Level 1 (spec-aware minify-html, conditional data-*, meta/link pruning) | Stage 3 |
| `src/stress/mutations.py` | Known-breaking + known-safe mutations | Stage 2.3 |
| `src/pipeline/00…03` | Runnable steps in order: download → determinism audit → Level 1 under the gate → stress test | Stages 1–3 |

The folder layout matches your old repo (`src/render`, `src/compress`,
`src/compare`, `src/pipeline`, `reports/csv`, `outputs`), so you can either
start clean from this kit or drop these files into your existing project.
**If you keep old files, delete or archive the old `render_html.py`,
`ssim_compare.py`, and `07_run_level2_condense.py` — they contain the bugs.**

---

## One-time setup (Windows PowerShell; macOS/Linux in parentheses)

```powershell
cd Desktop
mkdir token_efficient_ui_code ; cd token_efficient_ui_code
# unzip the starter kit contents into this folder, then:

python -m venv .venv
.venv\Scripts\Activate.ps1          # (source .venv/bin/activate)
pip install -r requirements.txt
playwright install chromium

git init ; git add -A ; git commit -m "starter kit: harness + gate v2"
```

Also create `lab_notebook.md` now and write today's date + "restarted project
on corrected harness/gate." Every day you work, add three lines: what you ran,
what surprised you, what you decided. This notebook is your defense evidence.

---

## Week 1, day by day

### Day 1 — data + determinism (Stage 1)
```powershell
python src/pipeline/00_download_pilot_data.py --source websight --n 100
python src/pipeline/01_determinism_audit.py  --source websight
```
**Done when:** ≥99/100 pages pixel-identical across double renders.
If a page flakes, open its `_a.png`/`_b.png` side by side; the usual culprits
are an animation that escaped the freeze or an asset type not intercepted.
Log the determinism rate in the notebook — it becomes one sentence in
Threats to Validity.

### Day 2 — Level 1 under the real gate (Stage 3)
```powershell
python src/pipeline/02_run_level1.py --source websight
```
**Done when:** you have `reports/csv/level1_gate_websight.csv` and can answer:
acceptance rate, median reduction, and — for any rejected page — *which gate*
tripped (the CSV has one boolean column per gate; that attribution is a paper
table, not debugging noise). Expect numbers near your old pilot (~10–14%
median) but now trustworthy.

### Day 3 — stress test v1 (Stage 2.3)
```powershell
python src/pipeline/03_run_stress_test.py --source websight --k 20
```
**Done when:** you can fill in this sentence with real numbers:
"The gate rejected __% of breaking mutants and accepted __% of safe mutants;
__% of breaking mutants had SSIM ≥ 0.95 and would have slipped an SSIM-only
gate." That last number is the headline of your metric-validation section.

### Day 4 — tune, re-run, widen
Read the per-gate columns of the stress CSV. If a breaking mutation class is
slipping through, the responsible threshold is too loose; if safe mutants are
being rejected, find which gate is too tight. Adjust `config/gate_config.yaml`,
re-run Day 3. Then pull the real-world stratum and repeat Days 1–3 on it:
```powershell
python src/pipeline/00_download_pilot_data.py --source webcode2m --n 100
python src/pipeline/01_determinism_audit.py  --source webcode2m
python src/pipeline/02_run_level1.py         --source webcode2m
python src/pipeline/03_run_stress_test.py    --source webcode2m --k 20
```
Real-world pages will be messier — some will crash or time out. That skip
rate is a *statistic to record*, not a failure to hide.

### Day 5 — freeze and commit (Stage 2 done)
When breaking-rejection ≥ 99% and safe-acceptance is high on **both** strata:
```powershell
git add config/gate_config.yaml ; git commit -m "FREEZE gate thresholds after stress test"
```
From this commit on, thresholds do not move mid-experiment. Write the
supervisor one paragraph: determinism rate, gate calibration numbers, Level-1
table on both datasets.

---

## Week 2 and beyond (where the protocol takes over)

1. **Rebuild Level 2** per protocol Stage 4 — reuse the `data-p2s-id`
   annotation idea from your old script, but on this harness, with the fixed
   visibility rules (no `aria-hidden`, no `<2px`, document-bounds not
   viewport), div/span-only collapsing, and batch-apply + bisection so every
   rejected edit is attributed to a rule. Level 2 is deliberately NOT in this
   kit: it must be built on top of the frozen gate, and its old numbers must
   not be trusted or reused.
2. **Level 3** (computed-style resynthesis) per protocol Stage 5, on the
   200-page calibration set (100 WebSight + 100 WebCode2M you now have).
3. **Scale** to 5,000 WebCode2M + 1,000 WebSight (protocol Stage 6), write the
   pre-registration (Stage 7), then fine-tune the four conditions (Stage 8).

---

## Troubleshooting

- **`ModuleNotFoundError: src`** — run scripts from the repo root (the folder
  containing `src/`), exactly as written above.
- **First `02_run_level1.py` run is slow** — it downloads the Qwen2.5-VL
  tokenizer once; if that fails (no internet/HF hiccup) it falls back to
  tiktoken automatically and tells you which counter it used.
- **Playwright errors about browsers** — re-run `playwright install chromium`
  inside the venv.
- **Want LPIPS (G6)?** `pip install torch lpips`, set `use_lpips: true` in the
  config. Leave it off until the stress test shows the hard gates need it.
- **HF download slow/blocked** — both datasets are public; retry, or reduce
  `--n` for a first smoke test (`--n 10` works end to end).

## Golden rules (tape above your monitor)
1. Never resize screenshots to compare them — a size mismatch is evidence.
2. `full_page=True`, always.
3. The gate decides acceptance; SSIM is a diagnostic column.
4. Thresholds move only before the freeze commit, never after.
5. Manifest CSVs define the dataset; folders are just storage.
6. Three lines in `lab_notebook.md` every working day.
