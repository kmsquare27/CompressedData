"""Step 03 -- the gate stress test (metric validation), v2.

Builds the labeled mutant benchmark that proves -- or falsifies -- the
efficacy of gates G0-G6, then hands the per-sample scores to step 04 for
ROC/AUC, threshold calibration and the freeze.

What changed vs v1 and why (full rationale in STRESS_TEST.md):

  * GROUND-TRUTH LABELS. Every mutant is render-verified against the
    original under the deterministic harness. Breaking mutants that change
    zero pixels are duds (target resampled up to --max-attempts, else
    excluded) -- never counted as gate misses. Safe mutants that DO change
    pixels are excluded from calibration and reported as findings.
  * VISUAL VERDICT ONLY. Detection is scored on G2-G6 (`accepted_visual`).
    v1 scored `accepted`, which includes G1 token reduction -- and since
    style-adding mutants (M3/M4/M5) INCREASE tokens, G1 alone "detected"
    them and the stress test could not fail. G0/G1 results are still
    recorded, they just don't decide.
  * SAME CODE PATH, CACHED ORIGINAL. Mutants are scored by
    src.compare.gate.evaluate_pair -- the exact production comparison code
    -- against ONE cached render of the stamped original per page (~2x
    fewer renders).
  * DETERMINISM. Seeds are CRC32 of (page, mutation, variant, attempt),
    not Python's per-process-salted hash(); a re-run reproduces every
    mutant byte-for-byte.
  * FULL PROTOCOL COVERAGE + SEVERITY. M1-M7 and S1-S6 all implemented,
    with graded severities (M3 8/16/32px, M4 dE00 2/5/10, M5 +/-2px), plus
    M8 (image deletion) and the X1 legacy-whitespace probe as marked
    extensions.

Usage:
    python src/pipeline/03_run_stress_test.py --source webcode2m --k 100
    # then: python src/pipeline/04_calibrate_gate.py --source webcode2m

Output: reports/csv/stress_samples_<source>.csv  (one row per mutant,
        resumable: re-running skips samples already present).
"""
from __future__ import annotations

import argparse
import sys
import time
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import random  # noqa: E402
from tqdm import tqdm  # noqa: E402

from src.compare.gate import (TokenCounter, evaluate_pair, load_config,  # noqa: E402
                              render_page, visual_accepted)
from src.render.harness import RenderHarness  # noqa: E402
from src.stress import annotate  # noqa: E402
from src.stress import mutations as MUT  # noqa: E402


def stable_seed(*parts) -> int:
    return zlib.crc32("|".join(map(str, parts)).encode("utf-8")) & 0xFFFFFFFF


def classify(intent: str, changed: bool):
    """(status, verified_label, calib_include) from intent x pixel truth."""
    if intent == "breaking":
        return ("ok", 1, True) if changed else ("dud", 0, False)
    if intent == "safe":
        return ("ok", 0, True) if not changed else ("unsafe_safe", 1, False)
    if intent == "tolerable":
        # Sub-JND: a NEGATIVE (must be accepted) that DOES change pixels --
        # the only kind of sample able to bound a threshold from above, since
        # the gate accepts identical rasters unconditionally. One that changed
        # nothing is not wrong, just uninformative: it collapses into the
        # pixel-identical safe class, so it is excluded rather than padding
        # the negative count with a sample that cannot be rejected.
        return ("ok", 0, True) if changed else ("tolerable_noop", 0, False)
    return ("ok", int(changed), False)  # probe: reported, never calibrated


def base_row(pid, source, spec, vkey):
    return {"sample_id": f"{pid}__{spec.name}__{vkey}", "page_id": pid,
            "source": source, "mutation": spec.name,
            "protocol_id": spec.protocol_id, "family": spec.family,
            "intent": spec.intent, "variant": vkey,
            "extension": spec.extension}


def run_source(source: str, args, harness, cfg, tk) -> Path:
    manifest = ROOT / "data" / "splits" / f"pilot_{source}_manifest.csv"
    if not manifest.exists():
        print(f"[03] no manifest for {source} ({manifest}); "
              f"run 00_download_pilot_data.py first"); return None
    df = pd.read_csv(manifest).head(args.k)

    stress_dir = ROOT / "outputs" / "stress" / source
    stress_dir.mkdir(parents=True, exist_ok=True)
    work = ROOT / "outputs" / "renders_gate" / source / "stress"
    work.mkdir(parents=True, exist_ok=True)
    rep = ROOT / "reports" / "csv"
    rep.mkdir(parents=True, exist_ok=True)
    out_csv = rep / f"stress_samples_{source}.csv"

    done = set()
    prior = None
    if out_csv.exists() and not args.no_resume:
        prior = pd.read_csv(out_csv)
        done = set(prior["sample_id"].astype(str))
        print(f"[03] resuming {source}: {len(done)} samples already scored")

    specs = [sp for sp in MUT.REGISTRY
             if not args.ops or sp.name in args.ops or sp.protocol_id in args.ops]

    rows = []

    def flush():
        nonlocal prior, rows
        if not rows:
            return
        new = pd.DataFrame(rows)
        prior = pd.concat([prior, new], ignore_index=True) if prior is not None else new
        prior.to_csv(out_csv, index=False)
        rows = []

    for _, r in tqdm(df.iterrows(), total=len(df), desc=f"stress[{source}]"):
        pid = str(r["page_id"])
        todo = [(sp, vk) for sp in specs for vk, _ in sp.variants
                if f"{pid}__{sp.name}__{vk}" not in done]
        if not todo:
            continue
        html = Path(r["html_path"]).read_text(encoding="utf-8", errors="ignore")

        if annotate.data_attr_selector_hazard(html):
            for sp, vk in todo:
                rows.append({**base_row(pid, source, sp, vk),
                             "status": "skipped_data_selector"})
            flush(); continue
        stamped, n_ids = annotate.stamp_ids(html)
        if stamped is None:
            for sp, vk in todo:
                rows.append({**base_row(pid, source, sp, vk),
                             "status": "skipped_no_body"})
            flush(); continue

        stamped_path = stress_dir / f"{pid}__stamped.html"
        stamped_path.write_text(stamped, encoding="utf-8")
        try:
            art_o = render_page(harness, stamped_path,
                                work / f"{pid}__orig.png", tk)
            meta = annotate.collect_meta(harness)   # stamped page still loaded
        except Exception as e:  # noqa: BLE001
            for sp, vk in todo:
                rows.append({**base_row(pid, source, sp, vk),
                             "status": f"page_error: {e}"})
            flush(); continue

        for sp, vk in todo:
            row = base_row(pid, source, sp, vk)
            exclude, outcome, art_c, pix, attempt = set(), None, None, None, 0
            try:
                for attempt in range(1, args.max_attempts + 1):
                    rng = random.Random(stable_seed(pid, sp.name, vk, attempt))
                    outcome = MUT.apply_mutation(sp.name, stamped, meta, rng,
                                                 vk, exclude)
                    if outcome is None:
                        break
                    mpath = stress_dir / f"{pid}__{sp.name}__{vk}.html"
                    mpath.write_text(outcome.html, encoding="utf-8")
                    art_c = render_page(harness, mpath,
                                        work / f"{pid}__{sp.name}__{vk}.png", tk)
                    pix = annotate.pixel_stats(art_o.img, art_c.img)
                    changed = annotate.visually_changed(pix)
                    resample = (sp.intent == "breaking" and not changed)
                    if sp.intent == "tolerable":
                        # Want small-but-nonzero. Nothing changed ->
                        # uninformative. Too much changed -> the operator did
                        # something other than intended (a 0.5px nudge that
                        # triggered a line re-wrap), and calling that
                        # imperceptible would loosen thresholds on a false
                        # premise. Cap is per-operator; ops bounded by
                        # construction set 0.
                        cap = getattr(sp, "max_diff_frac", 0.0)
                        resample = (not changed) or bool(
                            cap and pix["diff_frac"] > cap)
                    if (resample and outcome.target_ids
                            and attempt < args.max_attempts):
                        exclude |= set(outcome.target_ids)
                        continue
                    break
            except Exception as e:  # noqa: BLE001
                rows.append({**row, "status": f"failed: {e}"}); continue

            if outcome is None:
                # attempt-1 mutant renders happened before candidates ran out
                # (dud resampling); record them so render accounting is exact.
                rows.append({**row, "status": "na",
                             "attempts": max(0, attempt - 1)}); continue

            changed = annotate.visually_changed(pix)
            status, vlabel, calib = classify(sp.intent, changed)
            # Last-attempt overshoot: the resample loop can exhaust attempts
            # with the screen still violated. Such a sample is NOT a verified
            # imperceptible negative -- record it as a finding instead of
            # letting it widen a threshold.
            _cap = getattr(sp, "max_diff_frac", 0.0)
            if (sp.intent == "tolerable" and changed and _cap
                    and pix["diff_frac"] > _cap):
                status, vlabel, calib = "tolerable_reflow", 0, False
            row.update({"status": status, "attempts": attempt,
                        "target_ids": ";".join(map(str, outcome.target_ids)),
                        "severity_nominal": outcome.severity_nominal,
                        "severity_achieved": outcome.severity_achieved,
                        "detail": outcome.detail,
                        "verified_change": changed, "verified_label": vlabel,
                        "calib_include": calib, **pix})

            R = evaluate_pair(art_o, art_c, cfg,
                              page_id=f"{pid}__{sp.name}__{vk}")
            R.pop("page_id", None)
            row.update(R)
            row["accepted_visual"] = visual_accepted(R)
            row["accepted_full"] = R["accepted"]
            row["ssim_canvas"] = annotate.ssim_union_canvas(art_o.img, art_c.img)
            rows.append(row)
        flush()

    flush()
    return out_csv


def summarize(out_csv: Path, source: str) -> None:
    df = pd.read_csv(out_csv)
    print(f"\n[03] ---- stress test summary: {source} "
          f"({df['page_id'].nunique()} pages, {len(df)} samples) ----")
    print(df["status"].value_counts().to_string())

    ok = df[(df["status"] == "ok") & df.get("calib_include", False)].copy()
    if not len(ok):
        print("[03] no calibration-eligible samples yet"); return
    ok["accepted_visual"] = ok["accepted_visual"].astype(bool)
    brk = ok[ok["verified_label"] == 1]
    # By INTENT, not by label: tolerable samples are also verified_label == 0,
    # so selecting on the label would fold the sub-JND class into the safe
    # count and report each of them twice. The two classes are reported
    # separately because they mean different things -- safe is pixel-identical
    # and unrejectable, tolerable is pixel-different and is what actually
    # bounds a threshold.
    safe = ok[ok["intent"] == "safe"] if "intent" in ok else ok[ok["verified_label"] == 0]
    print(f"\nAt PROVISIONAL thresholds, visual gates G2-G6 only "
          f"(G0/G1 recorded but excluded -- see STRESS_TEST.md):")
    if len(brk):
        print(f"  verified-breaking rejected: "
              f"{100.0 * (~brk['accepted_visual']).mean():.1f}%  "
              f"({int((~brk['accepted_visual']).sum())}/{len(brk)}; target >= 99%)")
        per = brk.groupby(["protocol_id", "variant"])["accepted_visual"]
        print(per.apply(lambda s: f"{100.0 * (~s).mean():.0f}% rejected "
                                  f"(n={len(s)})").to_string())
    if len(safe):
        print(f"  verified-safe accepted:     "
              f"{100.0 * safe['accepted_visual'].mean():.1f}%  "
              f"({int(safe['accepted_visual'].sum())}/{len(safe)})")

    tol = ok[ok["intent"] == "tolerable"] if "intent" in ok else ok.iloc[:0]
    if len(tol):
        print(f"  sub-JND (tolerable) accepted: "
              f"{100.0 * tol['accepted_visual'].mean():.1f}%  "
              f"({int(tol['accepted_visual'].sum())}/{len(tol)})"
              f"   <-- the ONLY negatives that can constrain a threshold")
        per = tol.groupby(["protocol_id", "variant"])["accepted_visual"]
        print(per.apply(lambda s: f"{100.0 * s.mean():.0f}% accepted "
                                  f"(n={len(s)})").to_string())
        sev = pd.to_numeric(tol.get("severity_achieved"), errors="coerce")
        if sev.notna().any():
            print(f"  max achieved dE00 on T1: {sev.max():.2f} "
                  f"(JND ~= 1.0, so imperceptibility holds by construction)")

    if len(brk):
        for col, label in (("ssim_diag", "common-crop"), ("ssim_canvas", "union-canvas")):
            if col in brk:
                v = pd.to_numeric(brk[col], errors="coerce")
                print(f"  breaking mutants with {label} SSIM >= 0.95: "
                      f"{100.0 * (v >= 0.95).mean():.1f}%  "
                      f"<-- would slip an SSIM-only gate")

    finds = df[df["status"] == "unsafe_safe"]
    if len(finds):
        print(f"\nFINDINGS -- 'safe' edits that changed pixels "
              f"(excluded from calibration, reported):")
        print(finds.groupby("mutation").size().to_string())
    probe = df[(df["intent"] == "probe") & (df["status"] == "ok")]
    if len(probe):
        print(f"\nX1 legacy whitespace regex broke rendering on "
              f"{100.0 * probe['verified_change'].mean():.1f}% of pages "
              f"({int(probe['verified_change'].sum())}/{len(probe)}) "
              f"-- the case for minify-html, measured")
    print(f"\n[03] wrote {out_csv}")
    print("[03] NEXT: python src/pipeline/04_calibrate_gate.py --source webcode2m")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["webcode2m"], default="webcode2m")
    ap.add_argument("--k", type=int, default=100,
                    help="pages per source (protocol: 100 + 100)")
    ap.add_argument("--ops", nargs="*", default=None,
                    help="restrict to these mutation names or protocol ids")
    ap.add_argument("--max-attempts", type=int, default=3,
                    help="target resamples before a breaking no-op is a dud")
    ap.add_argument("--no-resume", action="store_true")
    args = ap.parse_args()

    cfg = load_config(ROOT / "config" / "gate_config.yaml")
    tk = TokenCounter.get()
    sources = [args.source]
    t0 = time.time()
    with RenderHarness() as h:
        for source in sources:
            out_csv = run_source(source, args, h, cfg, tk)
            if out_csv:
                summarize(out_csv, source)
    print(f"[03] total wall time: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()