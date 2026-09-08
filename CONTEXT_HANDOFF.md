# Project Context Handoff — Render-Validated HTML Compression

**Purpose:** carry the state of this work into a new conversation. Written for an
assistant with no prior context. Covers what the project is, what was decided and
why, what the numbers are, what is still open, and the traps that cost time.

**Thesis:** "Render-Validated HTML Compression for Screenshot-to-Code Fine-Tuning."
Fine-tuning target: Qwen2.5-VL. Dataset: WebCode2M (100-page pilot; ~12k planned,
possibly mixed with ScreenCoder later). Repo root:
`C:\Users\Mahin\Thesis\render_validated_starter_kit\starter_kit`.

---

## 1. What the project does

Screenshot-to-code datasets pair a webpage screenshot with its HTML. The HTML
targets are long, and much of their content does not affect the rendered page.
Compressing them cuts fine-tuning cost — but only if the compression provably
does not change what the page looks like, or the training label is silently
corrupted.

So every compressed candidate is **re-rendered and compared against the original
render** by a layered acceptance gate whose thresholds were calibrated by a
mutation stress test and then frozen. The contribution is the validation
apparatus as much as the compressor.

---

## 2. Pipeline as it stands

| Step | File | Purpose | State |
|---|---|---|---|
| 00 | `00_download_pilot_data.py` | build the page manifest | done |
| 01 | `01_determinism_audit.py` | prove same HTML → same pixels | done, 100/100 |
| 02 | `02_run_level1.py` | Level 1 (L1a + L1b), gated | done |
| 03 | `03_run_stress_test.py` | mutation benchmark | **done, do not re-run** |
| 04 | `04_calibrate_gate.py` | ROC, thresholds, freeze | **done, frozen** |
| 05 | `05_token_census.py` | where the tokens live | done |
| 06 | `06_run_level2.py` | Level 2 DOM condensation | done |
| 07 | `07_select_levels.py` | per-page best-level selection | v2 done; v3 pending |
| 08 | `08_final_validate.py` | re-gate selected vs original | **not run** |
| 09/10 | `09_css_census.py`, `10_run_level3.py` | Level 3 (CSS) | **not run, not decided** |

---

## 3. The acceptance gate (frozen — do not change)

`config/gate_config.yaml`:

```
height_tol_px: 1 · text_ratio_min: 0.9995 · iou_mean_min: 0.995
center_shift_max_px: 1.0 · deltae_p95_max: 0.5 · deltae_max: 1.0
require_token_reduction: true · use_lpips: false
```

Gates: G0 parse · G1 tokens strictly reduced · G2 document height · G3 visible
text (normalised edit ratio) · G4 visual blocks 1:1 with IoU and centre shift ·
G5 per-block CIEDE2000 · G6 LPIPS (disabled). **SSIM is recorded but never
gates.**

**Soundness rule** (in `gate.evaluate_pair`): if the two renders are
byte-identical, the page is visually equivalent and the structural gates cannot
veto. Every gate metric is DOM-derived and no DOM-derived quantity is a faithful
proxy for visual change — a transparent or same-coloured block can widen by
hundreds of pixels with zero pixel change.

### Calibration results

| policy | breaking recall | 95% CI | false rejection |
|---|---|---|---|
| provisional | 0.825 | [0.803, 0.845] | 0.002 |
| safe_max | 0.895 | [0.876, 0.911] | 0.000 |
| **strict (frozen)** | **0.992** | **[0.985, 0.995]** | **0.044** |

Calibration set: 1,206 verified-breaking / 564 verified-safe mutants over 96
pages. The 564 negatives are 380 pixel-identical + **184 sub-JND** (pixel-
different but below the perceptual JND; ΔE00 ≤ 0.65 against a JND of ~1.0). Only
the sub-JND samples can bound a threshold from above, because pixel-identical
ones are auto-accepted by the soundness rule.

**False rejection has two denominators, both correct:** 25 rejections over all
564 negatives = **4.4%**; over the 184 that could be rejected = **13.6%**. Say
which one you mean.

**R1 (pre-registered, SSIM sufficiency):** at an identical 4.4% false-rejection
rate, SSIM's best threshold (0.999) misses **31.9%** of verified-breaking
mutants; the layered gate misses **0.8%**. SSIM AUC 0.953.
**R2 (LPIPS value):** deferred — `use_lpips: false`, so G6 produced no column.

**Other stress findings:** 65 "safe" transformations changed pixels (S4 wrapper
collapse 29, S6 re-minify 26, S3 8, S5 1, S2 1). The naive whitespace regex (X1
probe) broke rendering on **54.2%** of pages. 3 duds.

---

## 4. Results

### Token census (100 pages, median 3,377 tokens)

CSS 39.6% · visible text 18.8% · tags 17.1% · class attrs 13.7% · other attrs
7.4% · inline styles 3.3% · whitespace 0.6% · **comments 0.0% · scripts 0.0%**.
148 elements/page, 26 wrapper candidates, mean depth 8.9.

WebCode2M ships pre-cleaned, which is why Level 1's original targets were nearly
empty. **CSS is the largest remaining mass and is still untouched.**

### Level 1 (current, after both bug fixes)

| | |
|---|---|
| L1a accepted | 97/98, **98% pixel-identical** |
| L1b accepted | 77/98, 77% pixel-identical |
| selected | **97/100** (l1b 76, l1a 21) |
| reduction on accepted | mean 13.79% |
| **corpus-level** | **13.37%** |

Progression across fixes: 4.60% mean / 3.08% corpus (broken) → 14.43% / 11.98%
(minify fixed, lxml) → **13.79% / 13.37%** (html5lib; acceptance 83 → 97).

The single remaining L1a rejection is `we00078` — a genuine lightningcss
non-neutrality (pixel-identical through every other step, then 16,082 pixels at
CSS minification). Correctly rejected.

### Level 2 (current)

| | |
|---|---|
| accepted | **90/100** (was 72) |
| reduction on accepted | mean 12.01%, median 8.55%, p90 24.98% |
| corpus vs each page's base | 10.81% |
| edits kept | **82.5%** (was 39.4%) |
| prescreen-dropped, no render | 259 edits |
| pages at the 40-call ceiling | **0** (was 42) |
| median gate calls/page | **1** (was 29) |
| runtime | 26 min (was 96) |
| savings split | invisible subtrees 75.2% / wrapper collapse 24.8% |

### Combined (computed from the Level 2 CSV; step 07 v3 + 08 will confirm)

On the 90 Level-2-accepted pages: original 356,156 tokens → 265,100 =
**25.57%**, of which L1 contributes 14.73 points and L2 adds 10.84.
Corpus lower bound counting the other 10 pages as zero: **24.12%**.
Expected after selection: **~24.9%**.

Trajectory: 3.08% → 9.11% → **~25%**, with the gate unchanged throughout.

---

## 5. Decisions taken (and who decided)

| Decision | Rationale |
|---|---|
| **Do not build Level 3 as page-rewriting** (user) | Rebuilding pages into a synthesised style dialect means the model learns to emit that dialect, not real HTML. Levels 1–2 stay subtractive. A *subtractive* CSS level (delete dead rules) remains open. |
| **Keep the gate frozen; never re-run step 03** | Thresholds were fixed by an independent stress test before any compression was measured. Re-deriving them after seeing results destroys that ordering. Later changes to `annotate.py` (parser, hazard rule, `paintable`) mean a re-run would generate different mutants. |
| **Every page stays in the dataset** (user) | Rejection means a less-compressed target, never a dropped page. Four cases: L1✓L2✓ → level2 (stacked); L1✓L2✗ → level1; L1✗L2✓ → level2 built from original; neither → original at 0%. |
| **Select on absolute tokens, not `reduction_pct`** | The gate CSVs measure percentages against different bases (L2's is relative to whatever it composed on), so only absolute counts are comparable. |
| **No minimum-reduction threshold** | Rejecting small wins sends those pages to 0%; it raises the per-page average while lowering corpus savings. |
| **Page selection may use size/source, never compression ratio** | Filtering on the outcome is circular and a reviewer will catch it. |
| **Report per-page reduction only alongside acceptance rate** | Otherwise it reads as conditioning on the outcome. Corpus-level (rejections at 0%) is the honest headline. |
| **`--budget 80` rejected** | Measured on the same 20 pages: 7.66% at budget 40 vs 7.12% at 80. Budget was never the constraint; the prescreen later reduced median gate calls to 1. |
| **Stop chasing sub-1% artifacts** (user) | At 12k pages a 0.5% edge case is noise. Applies to the `we00007` ΔE artifact. |
| **`bisect_l1a.py` keeps lxml** | It is a forensic tool whose job is reproducing the old behaviour. |

---

## 6. Bugs found and fixed — each cost real time

1. **G1 contamination.** The v1 stress test scored mutants with the *full* gate.
   Style-adding breaking mutants (M3/M4/M5) increase tokens, so G1 rejected them
   before any visual gate ran; the test reported near-perfect detection for a
   bookkeeping reason and **could not fail**. Fixed: the stress verdict uses
   visual gates G2–G6 only.

2. **Unverified labels.** Intent was trusted as ground truth. Fixed by pixel-
   verifying every mutant: breaking-but-unchanged → dud (resampled, excluded);
   safe-but-changed → `unsafe_safe` finding (excluded, reported).

3. **DOM geometry ≠ visual geometry.** G4 measured `getBoundingClientRect()`. A
   transparent block can widen 100px → 1064px with zero pixel change, moving its
   centre 482px. Every safe false-rejection in the 87-page run was this, and it
   set the zero-FPR bar so high (centre shift 482px, ΔE 10.5) that per-metric
   calibration collapsed. Fixed by ink rects **and** the soundness rule.

4. **Degenerate negative class.** All safe mutants were pixel-identical, so no
   threshold could ever be bounded from above and calibration defaulted to
   floors. Fixed by adding the sub-JND (`tolerable`) class.

5. **Step 04 modelled a gate that no longer existed** — it applied thresholds to
   raw metrics without the soundness rule, reporting 1.05% false rejection where
   the shipped gate gives 0%.

6. **Composing on rejected output.** Step 02 writes `outputs/level1/<id>.html`
   for every page *before* gating, so step 06 would stack Level 2 on known-broken
   Level 1 output and validate against that broken baseline. Fixed: step 06
   consults the Level 1 gate CSV; step 02 now unlinks on rejection.

7. **SSIM memory.** A 100-page run died at page 87 — scikit-image promotes uint8
   to float64 and holds ~16 full-size temporaries, so peak memory scaled with
   page height (~465 MB at 1280×2836). Replaced with a strip-wise computation:
   bit-exact to 7 decimals, peak pinned at ~42 MB, 6–11× faster.

8. **Silent `minify_html` fallback.** `level1_minify.py` passed
   `do_not_minify_doctype`, renamed to `minify_doctype` in minify-html ≥0.16. The
   `TypeError` fell through to `minify_html.minify(out)` with **all defaults** —
   `minify_css=False`. **CSS minification had never run.** The reported 4.60% came
   from tag omission, not CSS. Fixed; a wrong kwarg now raises.

9. **The lxml round-trip.** *The biggest finding.* A no-op parse-and-serialize
   (no transforms at all) changed the render on **14 of 15** failing pages. lxml
   repairs malformed HTML — inserting `html`/`head`/`body`, relocating content out
   of `<head>`, restructuring `form`-in-`table` — by rules that do not match
   Chromium's. Measured faithfulness on those pages: lxml **1/15**, html.parser
   12/15, **html5lib 14/15**. html5lib implements the WHATWG algorithm browsers
   use. Fixed by switching all six `BeautifulSoup(...)` call sites to a shared
   `HTML_PARSER = "html5lib"`. L1a pixel-identical 85% → 98%, acceptance 83 → 97.
   Speed is a non-issue (12.2 ms vs 13.2 ms per page).

10. **`get_text()` returns `''` for `<style>` under html5lib.** Only the lxml and
    html.parser builders wrap stylesheet text in a `Stylesheet` node; html5lib
    emits a plain `NavigableString`, which `get_text()` filters out because a
    `<style>` tag's `interesting_string_types` is `Stylesheet`. This silently
    disabled every `[data-` guard — `level1a` would have stripped `data-*` on
    exactly the pages whose CSS selects them. Fixed with a `style_text()` helper
    reading `.contents`. **The write path was separately verified safe:**
    `Stylesheet(new)` serialises unescaped under html5lib.

---

## 7. Architecture worth knowing

**Level 2 bisection.** ~30% of individual neutral-wrapper collapses change pixels
(margin collapsing, unpredictable from computed style). A page with ten wrappers
therefore almost never passes as a batch, so on rejection the runner splits the
edit list, tests halves, recurses, blacklists the minimal offending subset and
keeps the rest. Every reported set is gate-verified **as a whole** — two
independently-safe halves need not be safe together.

**The prescreen** (added later) applies each edit to the live DOM, compares
computed styles, border-box rects, pseudo-element content and scroll size via an
in-page oracle, then restores. Edits indistinguishable from a no-op skip the
render gate entirely. This is what took median gate calls from 29 to 1. Failure
mode is safe: a false negative drops a good edit; survivors are still render-gated.

**Level 2 gates against the ORIGINAL render**, not the stamped base, so L1 and L2
tolerances cannot accumulate. `tokens_orig` stays relative to the base (G1
semantics); `tokens_original` is recorded separately for step 07.

**Level 1 is two stages.** L1a: comments, scripts, non-stylesheet `<link>`, most
`<meta>`, `on*`/`aria-*`/`title`/`data-*` attributes, and **CSS-only**
minification via lightningcss reached by wrapping the bare stylesheet in
`<style>` so minify-html never sees the HTML. L1b: minify-html over the whole
document with CSS off. L1b is expected to fail where minify-html deletes
whitespace between inline children of layout tags — median centre shift on L1b
rejections is 4.0px, one space at 14–16px type. Both stages are gated against the
original; the fewest-token accepted stage wins.

---

## 8. Open items

**Immediate:** step 07 v3 (needs a one-line fix — see below) then step 08, which
re-gates each selected artifact against a fresh render of the original and
reports how many are pixel-identical.

**Step 07 v3 bug (must fix before use):** in `load_level`,
`d.get("base", "original")` returns a *string* when the column is missing, so
`.where("original" == "original")` raises `ValueError: Array conditional must be
same shape as self` on the Level 1 CSV. Fix:

```python
base_col = (d["base"].astype(str) if "base" in d.columns
            else pd.Series("original", index=d.index))
```

and use `base_col` in both the `.where(...)` and the `f"{level}_base"` field.
Verified: crashes before, all four selection cases correct after. v3 is required
only if running step 08 (it emits the `ladder` column and `dataset_png_path` that
08 consumes).

**Undecided — CSS.** 39.6% of tokens, untouched. The subtractive option is
deleting rules dead under the training render: selectors matching nothing,
`@media` false at 1280px, `:hover`/`:focus` (no pointer exists), `@keyframes`
(animations frozen), `@font-face` with remote `src` (blocked). Step 09 measures
the prize before building. **Framing decision required:** removing `@media` and
`:hover` means the model never learns to emit them. Defensible under "reproduce
this screenshot"; must be stated as a scope limitation, not discovered by a
reviewer.

**Known limitations to state in the paper:**
- Calibration and evaluation share the same 96 pages.
- The gate was validated for structural damage, not style-detail damage — G4
  measures boxes and G5 is a per-block *mean* ΔE, so a deleted `border-radius`,
  `box-shadow` or `letter-spacing` moves neither. No M-operator covers these, so
  99.2% recall does not extend to CSS edits. This is why any CSS level should
  require pixel identity rather than the tolerance gate.
- The training manifest carries the *dataset's* screenshot while equivalence is
  proven under the harness (placeholder images, blocked remote assets, 1280px).
  Step 08 can switch to harness renders — but that changes the input distribution
  and is a decision, not a fix.
- Removing `display:none` subtrees means the target no longer contains hidden
  menus. The screenshot is unchanged so the pair stays consistent, but the model
  will not learn to emit hidden content.
- Sequence-level accounting: WebCode2M is ~65% image tokens, ~32% code, so ~25%
  of code tokens is ~8% of the training sequence. Report both.
- `we00053` and `we00071` time out at 30s on every run — a ~2% render-failure
  rate worth reporting rather than hiding.
- **The central hypothesis is untested.** No fine-tuning experiment exists yet.

---

## 9. Operational traps

- **Deleting the stress CSV is required after any change to `gate.py` or
  `harness.py`** — resume keys on `page_id__mutation__variant` and has no notion
  of code version, so it will silently re-report stale numbers. A run that
  finishes in seconds resumed; a real run takes minutes.
- **`06 --k N` overwrites the whole CSV** with just those N pages. This destroyed
  a 100-page result once. Archive first.
- **Step 04 overwrites everything** — `gate_calibration.csv`, all three figures,
  `stress_decisions.md`, `gate_config.calibrated.yaml` — with fixed filenames.
- **`--freeze` writes `config/gate_config.yaml`.** Do not pass it again.
- After changing `level1_minify.py`, clear `outputs/level1*/` so stale artifacts
  for now-rejected pages cannot be picked up.
- All six `BeautifulSoup(...)` call sites must use the same parser. If
  `stamp_ids` and `apply_edits` disagree, Level 2 re-introduces round-trip damage
  between stamping and editing.

---

## 10. Key citations

**Task/datasets:** Design2Code (Si et al., 2024) — the direct ancestor of the
gate's block/text/colour/position metrics and its Jonker-Volgenant matching;
WebCode2M (Gui et al., WWW 2025); WebSight (Laurençon et al., 2024); pix2code
(Beltramelli, 2018); ScreenCoder (2025).

**Closest prior work:** Pix2Struct (Lee et al., ICML 2023) — pretrains by parsing
screenshots into *simplified* HTML, but the simplification is a fixed recipe with
no per-example render validation; its own paper calls the trade-off "reasonable"
without measuring it. EfficientUICoder (Xiao et al., FSE 2026) — compresses at
inference, not in the training data, no re-render; reports 41.4% generated-token
reduction, a number to address head-on.

**Metrics:** SSIM (Wang et al., 2004); LPIPS (Zhang et al., CVPR 2018 — supports
R1: PSNR/SSIM are "simple, shallow functions"); CIEDE2000 (Luo et al. 2001 /
Sharma et al. 2005).

**Methodology:** mutation testing — DeMillo et al. 1978; Jia & Harman 2011; Just
et al. FSE 2014 ("are mutants a valid substitute for real faults"); the
equivalent-mutant problem maps directly onto duds. Metamorphic testing (Chen et
al. 1998) for the safe operators. Wilson (1927) for CIs; Fawcett (2006) for ROC.

**Must engage with:** "The Hidden Cost of Readability: How Code Formatting
Silently Consumes Your LLM Budget" (Pan et al., ICSE 2026, arXiv 2508.13666) —
essentially the Level 1 argument, published. Differentiator: formatting
compression for plain code has no correctness oracle; this work has one.

---

## 11. Framing

Corpus reduction is ~25% with CSS untouched, so the **token-efficiency** framing
is viable: a validated gate plus real reduction on real-world pages, every page
retained. The fallback, if CSS work stalls, is a **methods** contribution — a
validated acceptance gate plus the measurement of how little safe compression
real-world HTML admits. Assets for either: 99.2% recall at a frozen threshold,
the SSIM kill-shot at matched operating points, 65 "safe" transformations that
provably were not, a whitespace regex that breaks 54% of pages, and the finding
that what compresses is hidden content rather than redundant structure.

Two findings generalise beyond this project and are worth stating plainly:
**serialization is not neutral** — a DOM parse-and-serialize round trip changed
the render on 14% of real-world pages under lxml — and **the compressible mass in
real HTML is content that never renders**, not redundant markup.
