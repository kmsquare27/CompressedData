# Handoff — Render-Validated HTML Compression (WebCode2M pilot)

## What this project is

Screenshot-to-code fine-tuning needs `(screenshot, HTML)` pairs. The HTML targets carry
tokens that never appear in the render. This pipeline shortens the HTML and proves, by
rendering it back, that the page still looks the same. Every compressed target is
accepted only if it reproduces the original's screenshot under one fixed harness
(Chromium, 1280 px, full page, images → 2×2 gray placeholder, remote assets blocked,
animations frozen).

Repo: `C:\Users\Mahin\Thesis\render_validated_starter_kit\starter_kit`
Layout: `src/compress`, `src/pipeline`, `src/stress`, `src/render`, `src/compare`, `src/tools`

## Current validated result (before the reporting fixes re-run)

| | |
|---|---|
| Corpus token saving | **30.84 %** across 100 input pages (2 timed out, counted at 0 %) |
| Validated pages | 98/100, every selected target re-validated against the original |
| Fallbacks | 0 — the selected artifact passed as-is 98/98 |
| Pixel-identical | 94/98 |
| Level mix | L3 84, L2 12, L1 2 |
| Tokenizer | Qwen2.5-VL |

The four non-pixel-identical pages are being handled by making pixel identity the
default in step 08 (they fall back a rung instead of being dropped).

## Pipeline

| Step | Does | Status |
|---|---|---|
| 00 download → 01 determinism → 02 L1 → 03 stress → 04 calibrate → 05 token census | setup, gate calibration | done |
| 06 L2 (DOM condensation) | removes invisible subtrees, collapses neutral wrappers; in-page prescreen then render-gated bisection | done, 72→ higher after fixes |
| 07 select | per-page ladder of accepted artifacts, sorted by tokens | **re-run needed** |
| 08 final validate | re-gates the selected artifact against the original, falls back a rung, freezes targets, writes the training manifest | **re-run needed** |
| 09 CSS census | sizes dead CSS before building L3 | done (9.4 % candidates) |
| 10 L3 (dead CSS) | removes unmatched / stateful / dead-property CSS; accepts on pixel identity | done: 84/100 accepted, 25,495 tokens, 94 % of the census ceiling |

Diagnostics in `src/tools/`: `diag_l2_rerun.py`, `find_l2_culprit.py`, `find_culprit2.py`.

## Findings worth keeping for the write-up

- **Where the savings are.** L2: 88 % of its savings from invisible subtrees. L3: 52 %
  unmatched selectors + 42 % dead selector-list items. One claim measured twice — what
  compresses in real UI code is content and style written for states and elements that
  are not in the captured page.
- **Size gradient.** 21 % on the smallest quartile → 37.7 % on the largest.
- **The gate's blind spot, measured.** 3 pages carry a visible change (up to 1162 px,
  channel delta 190) that the tolerance gate accepted with ΔE 0.000 and IoU 1.000.
  Strict pixel identity caught them. Culprits: empty `<span>` elements
  (`<span id="more-16221"></span>`, `<span class="author vcard"></span>`) removed as
  invisible but painting through generated content — the annotator's `paintable` test
  measures the element's own box, not its pseudo-elements.
- **`we00091`**: the difference is already present at L1, and no single L1 operator
  reproduces it in isolation (an interaction). Not chased further.
- **L1 is not render-neutral by construction.** Counterexamples: `[aria-hidden]`
  selectors, `content: attr(title)`, DOM-writing inline scripts. L1a *proposes*
  transformations; the gate checks them.
- **The oracle is a prefilter, not a proof.** For a removal it allows the removed
  subtree's own keys to vanish, so a misclassified-invisible element that paints slips
  through. That is why the render gate stays the arbiter.
- **Two acceptance contracts.** Strict pixel identity decides the dataset; the
  calibrated multi-metric gate is a diagnostic and a separate tolerance experiment.
  State this once, up front.

## Just-completed code changes (10 files, compiled, not yet run)

07 base-label normalization + L1a/L1b ladder rungs, `--write-final-manifest` removed ·
08 pixel identity by default (`--tolerance` opts out), tokens counted from file, exact
corpus denominator, fallback count fixed, frozen copy validated at its final path,
output dirs cleared, harness/gate hashes recorded · 02/06 "page-average" relabel + true
corpus line + `--report-only` · `level1_minify.py` docstring retraction · `gate.py` G5/G6
fail closed, no tokenizer fallback · 04 width check in `composite_reject`, rounded
thresholds evaluated, `--freeze` refuses on missed target, new `--evaluate-frozen` ·
`mutations.py` S2 order-preserving formatter · 03 `html_changed` / `noop` ·
`harness.py` goto timeout 30 s → 90 s.

None of these invalidate the rendered artifacts, so **02, 06 and 10 do not need re-running.**

---

# WHAT TO RUN NOW

Housekeeping first (the `.pyc` files are tracked; drop them before tagging):

```
echo __pycache__/ >> .gitignore
echo *.pyc >> .gitignore
git rm -r --cached "**/__pycache__"
```

Back up the current results, then delete the diagnostic scratch:

```
mkdir runs\2026-09-08_pre_fix
xcopy /E /I reports\csv runs\2026-09-08_pre_fix\csv
xcopy /E /I data\splits runs\2026-09-08_pre_fix\splits
xcopy /E /I outputs\final_targets runs\2026-09-08_pre_fix\final_targets

del data\splits\pilot_webcode2m_diag_manifest.csv
del reports\csv\*webcode2m_diag*.csv reports\csv\diag_l2_rerun_webcode2m.csv reports\csv\l2_culprit_webcode2m.csv reports\csv\culprit2_webcode2m.csv
rmdir /S /Q outputs\level1\webcode2m_diag outputs\level2\webcode2m_diag outputs\renders_gate\webcode2m_diag
rmdir /S /Q outputs\renders_gate\webcode2m\culprit outputs\renders_gate\webcode2m\culprit2 outputs\renders_gate\webcode2m\diag
```

Then, in order:

```
:: 0  sanity — expect three zeros (no gate errors in the runs being kept)
python -c "import pandas as pd; [print(f, pd.read_csv(f).get('g5_error', pd.Series(dtype=str)).notna().sum()) for f in ['reports/csv/level1_stages_webcode2m.csv','reports/csv/level2_gate_webcode2m.csv','reports/csv/level3_gate_webcode2m.csv']]"

:: 1  corrected L1/L2 report lines — seconds, no renders
python src/pipeline/02_run_level1.py --source webcode2m --report-only
python src/pipeline/06_run_level2.py --source webcode2m --report-only

:: 2  calibration metrics of the policy that actually ran — seconds, no renders
python src/pipeline/04_calibrate_gate.py --source webcode2m --evaluate-frozen

:: 3  selection with the full ladder — seconds
python src/pipeline/07_select_levels.py --source webcode2m

:: 4  final validation, strict by default — ~8 min
python src/pipeline/08_final_validate.py --source webcode2m --repeat

:: 5  verify the manifest points at files that exist — expect "missing files 0"
python -c "import pandas as pd, pathlib; m=pd.read_csv('data/splits/compressed_webcode2m_manifest.csv'); missing=[p for p in m.html_path if not pathlib.Path(p).exists()]; print('rows', len(m), 'missing files', len(missing)); print(missing[:5])"

:: 6  freeze
git add -A && git commit -m "dataset v1: strict pixel-identity validation, corrected reporting"
git tag dataset-v1
```

## What each command rewrites

| Command | Writes / overwrites | Leaves alone |
|---|---|---|
| 02 `--report-only` | **nothing** (prints only) | all |
| 06 `--report-only` | **nothing** (prints only) | all |
| 04 `--evaluate-frozen` | `reports/csv/gate_calibration_frozen_webcode2m.csv` (new) | `config/gate_config.yaml`, `gate_calibration.csv`, stress CSVs |
| 07 | `reports/csv/level_selection_webcode2m.csv`, `data/splits/selected_webcode2m_manifest.csv` | all outputs, all gate CSVs |
| 08 `--repeat` | `outputs/final_targets/webcode2m/*` and `outputs/renders_final/webcode2m/*` (**cleared first**), `outputs/renders_gate/webcode2m/final/*`, `reports/csv/final_validation_webcode2m.csv`, `reports/csv/final_validation_webcode2m_environment.json`, `data/splits/compressed_webcode2m_manifest.csv` | `outputs/level1*`, `level2`, `level3`, all level gate CSVs, stress CSVs, config |

Never deleted or re-run: `level1_gate`, `level1_stages`, `level2_gate`, `level2_edits`,
`level3_gate`, `stress_samples`, `gate_calibration`, and everything under
`outputs/level1a`, `level1b`, `level1`, `level2`, `level3`. Those are the validated
inputs the re-run consumes.

## Checkpoints while running

- **After 07** — the composition table must show ~90 under "L1 + L2 stacked" (it was
  missing before the label fix), and the `ladder` column must contain `l1a` rungs.
- **After 08** — `pixel-identical to original` must equal the number of `ok` pages.
  The exact all-inputs corpus figure is the thesis number (expect slightly below
  30.84 %: the 4 non-identical pages now fall back a rung). With the 90 s timeout,
  expect 100 `ok` instead of 98. Watch `final_level` for any page landing on `l1a` —
  that is the ladder fix earning its keep.

## The dataset, once 08 finishes

| What | Where |
|---|---|
| Manifest (start here) | `data/splits/compressed_webcode2m_manifest.csv` |
| Compressed HTML targets | `outputs/final_targets/webcode2m/<page_id>.html` |
| Training screenshots | `outputs/renders_final/webcode2m/<page_id>.png` |
| Audit trail | `reports/csv/final_validation_webcode2m.csv` |

Columns: `png_path`, `html_path`, `level`, `tokens_orig`, `tokens_final`,
`reduction_pct`, `original_sha256`, `final_sha256`, `validated_source_path`.

Use `png_path` from the manifest, **not** the dataset's own screenshot — equivalence was
proven under the harness. For the baseline arm, pair the *same* `png_path` with the
original HTML (`pilot_webcode2m_manifest.csv` → `html_path`). Same images in both arms;
only the targets differ.

## Next milestone: fine-tuning

Two arms, identical images and identical page set:
control = (harness render → original HTML), treatment = (harness render → compressed HTML).
Pages where compression was rejected keep the original target in *both* arms, so the
compressed condition does not benefit from a different page population.
Report token/compute savings and reconstruction quality separately.

## Deferred (do while the GPU is busy; none of it changes the dataset)

- Stress test in one pass with `--no-resume`: S2 fix + `html_changed`/`noop`, three
  missing controls (raw vs stamped original, stamped vs no-edit reserialization,
  repeated baseline render), and the absent style-detail mutation families
  (border-radius, box-shadow, text-decoration, pseudo-element content, background-image)
  — the families the two `<span>` culprits and L3's operators actually touch.
- Calibration and evaluation on disjoint page sets; page-level resampling for CIs.
- `09_css_census.py --n 30 --ablate --merge` on the post-L3 targets to size the
  second-order prize and settle whether a Level 3b exists.
- Scale beyond the 100-page pilot.
