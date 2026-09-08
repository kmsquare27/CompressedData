"""Step 04 -- calibrate the gate from the stress-test benchmark and FREEZE it.

Consumes reports/csv/stress_samples_<source>.csv (step 03) and produces the
protocol's Stage-2 Definition of Done:

  reports/csv/gate_calibration.csv     per-metric ROC/AUC + operating points,
                                       composite recall/FPR under each policy
  reports/figures/roc_grid.png         one ROC per metric
  reports/figures/detection_heatmap.png  mutation-family x metric detection
  reports/figures/severity_curves.png  detection vs severity (M3/M4/M5)
  config/gate_config.calibrated.yaml   calibrated thresholds + provenance
  reports/stress_decisions.md          the two PRE-REGISTERED decision rules
                                       (R1 SSIM kill-shot, R2 LPIPS marginal
                                       value), findings, and the freeze
                                       recommendation
  --freeze                             writes config/gate_config.yaml

Calibration uses VERIFIED labels only (calib_include == True): duds and
unsafe-safe findings never contaminate recall/FPR. Detection is defined on
the visual metrics; G1 tokens and G0 parse are excluded by construction
(see STRESS_TEST.md for why leaving G1 in fakes a perfect gate).

Threshold policies evaluated side by side:
  provisional  the current gate_config.yaml values (asserted, not calibrated)
  safe_max     per metric: max score observed on the VERIFIED-SAFE set,
               times --slack, floored at the metric's measurement floor.
               Zero safe false-rejections on calibration data by
               construction; achieved breaking recall is then MEASURED.
  strict       measurement floors only (tightest defensible).
The frozen pick is the qualifying policy (composite recall >= --target)
with the loosest thresholds; if none qualifies, the best one is reported
with the slipping mutation classes named, so the gap is actionable.

Usage:
    python src/pipeline/04_calibrate_gate.py
    python src/pipeline/04_calibrate_gate.py --freeze
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.compare.gate import load_config  # noqa: E402

# ---------------------------------------------------------------------------
# Metric definitions: continuous "badness" scores (higher = more different),
# derived from the SAME columns the production gate computes. A metric FAILS
# a sample when score > threshold. n_unmatched > 0 is structural (any
# omission/fabrication fails G4 outright, no threshold to calibrate).
# ---------------------------------------------------------------------------
METRICS = [
    # name, derive(df)->Series, gate, cfg_key, thr->cfg transform, floor
    ("height_delta", lambda d: pd.to_numeric(d["height_delta_px"], errors="coerce"),
     "G2", "height_tol_px", lambda t: int(np.ceil(t)), 1.0),
    ("text_dissim", lambda d: 1.0 - pd.to_numeric(d["text_ratio"], errors="coerce"),
     "G3", "text_ratio_min", lambda t: round(1.0 - t, 4), 5e-4),
    ("iou_deficit", lambda d: 1.0 - pd.to_numeric(d["iou_mean"], errors="coerce"),
     "G4", "iou_mean_min", lambda t: round(1.0 - t, 4), 0.005),
    ("center_shift", lambda d: pd.to_numeric(d["center_shift_max"], errors="coerce"),
     "G4", "center_shift_max_px", lambda t: round(float(t), 1), 1.0),
    ("deltae_p95", lambda d: pd.to_numeric(d["deltae_p95"], errors="coerce"),
     "G5", "deltae_p95_max", lambda t: round(float(t), 2), 0.5),
    ("deltae_max", lambda d: pd.to_numeric(d["deltae_max"], errors="coerce"),
     "G5", "deltae_max", lambda t: round(float(t), 2), 1.0),
    ("lpips_max_tile", lambda d: pd.to_numeric(d["lpips_max_tile"], errors="coerce")
     if "lpips_max_tile" in d else pd.Series(np.nan, index=d.index),
     "G6", "lpips_max", lambda t: round(float(t), 3), 0.005),
]

SSIM_SLIP_THRESHOLDS = (0.90, 0.95, 0.99, 0.995, 0.999)


def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def roc(scores: np.ndarray, labels: np.ndarray):
    """Hand-rolled ROC (detect when score > threshold). Returns
    (fpr, tpr, thresholds, auc)."""
    m = ~np.isnan(scores)
    s, y = scores[m], labels[m]
    if len(s) == 0 or y.sum() == 0 or (1 - y).sum() == 0:
        return None
    order = np.argsort(-s, kind="mergesort")
    s, y = s[order], y[order]
    P, N = int(y.sum()), int(len(y) - y.sum())
    tps, fps = np.cumsum(y), np.cumsum(1 - y)
    idx = np.where(np.diff(s, append=-np.inf) != 0)[0]
    tpr = np.concatenate([[0.0], tps[idx] / P])
    fpr = np.concatenate([[0.0], fps[idx] / N])
    thr = np.concatenate([[np.inf], s[idx]])
    trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))
    return fpr, tpr, thr, float(trapz(tpr, fpr))


def max_recall_at_fpr(fpr, tpr, cap: float) -> float:
    ok = tpr[fpr <= cap + 1e-12]
    return float(ok.max()) if len(ok) else 0.0


def metric_scores(df: pd.DataFrame) -> dict:
    return {name: fn(df).astype(float) for name, fn, *_ in METRICS}


def pixel_identical_mask(df: pd.DataFrame) -> pd.Series:
    """Samples whose two renders are byte-identical.

    gate.py short-circuits these to ACCEPT: identical rasters mean identical
    appearance, and every gate metric is DOM-derived, so a metric can move
    when nothing on screen did (a transparent or same-coloured block widening
    over empty space shifts its box centre by hundreds of pixels while the
    render is unchanged). Calibration has to model the gate that actually
    ships, so the same rule is applied here. Prefers the gate's own
    `pixel_identical` column and falls back to the runner's pixel diff for
    CSVs written before that column existed."""
    if "pixel_identical" in df.columns:
        m = df["pixel_identical"]
        if m.notna().any():
            return m.astype(str).str.lower().isin(["true", "1"])
    if "n_diff_pixels" in df.columns:
        return pd.to_numeric(df["n_diff_pixels"], errors="coerce").fillna(-1) == 0
    return pd.Series(False, index=df.index)


def composite_reject(df: pd.DataFrame, thr: dict, use_lpips: bool) -> pd.Series:
    """Re-evaluate the layered visual gate at arbitrary thresholds."""
    S = metric_scores(df)
    rej = pd.to_numeric(df["n_unmatched"], errors="coerce").fillna(0) > 0
    for name, *_ in METRICS:
        if name == "lpips_max_tile" and not use_lpips:
            continue
        v = S[name]
        rej = rej | (v > thr[name]).fillna(False)
    # G2 also rejects on a WIDTH mismatch (gate.py never resizes screenshots);
    # none of the per-metric scores above are width-sensitive (height_delta is
    # height-only), so it has to be pulled in separately.
    if "shape_mismatch" in df.columns:
        rej = rej | df["shape_mismatch"].astype(str).str.lower().isin(["true", "1"])
    # Soundness rule (see gate.evaluate_pair): an unchanged raster cannot be
    # rejected, whatever the DOM-derived metrics say. Never suppresses a real
    # detection -- a mutant that changed zero pixels broke zero pixels.
    return rej & ~pixel_identical_mask(df)


def thresholds_from_cfg(cfg: dict) -> dict:
    return {"height_delta": cfg["height_tol_px"],
            "text_dissim": 1.0 - cfg["text_ratio_min"],
            "iou_deficit": 1.0 - cfg["iou_mean_min"],
            "center_shift": cfg["center_shift_max_px"],
            "deltae_p95": cfg["deltae_p95_max"],
            "deltae_max": cfg["deltae_max"],
            "lpips_max_tile": cfg["lpips_max"]}


def cfg_from_thresholds(thr: dict, base_cfg: dict) -> dict:
    out = dict(base_cfg)
    for name, _fn, _gate, cfg_key, to_cfg, _floor in METRICS:
        out[cfg_key] = to_cfg(thr[name])
    return out


def sha256(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()[:16]
    except Exception:
        return "n/a"


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "n/a"


def _evaluate_frozen(calib: pd.DataFrame, y: np.ndarray, cfg: dict, use_lpips: bool,
                     n_brk: int, source: str, rep: Path):
    """Recall/FPR/per-family recall of config/gate_config.yaml AS-IS (no
    rounding round-trip needed: values read straight from the frozen YAML
    already ARE the saved representation) on the existing stress CSV."""
    thr = thresholds_from_cfg(cfg)
    rej = composite_reject(calib, thr, use_lpips)
    rec = float(rej[y == 1].mean())
    fpr_ = float(rej[y == 0].mean())
    lo, hi = wilson(int(rej[y == 1].sum()), n_brk)
    row = {"kind": "composite", "name": "frozen", "source": source,
           "recall": round(rec, 4), "recall_ci_lo": round(lo, 4),
           "recall_ci_hi": round(hi, 4), "fpr": round(fpr_, 4),
           "thresholds": json.dumps({k: round(float(v), 5) for k, v in thr.items()})}
    for fam, g in calib[y == 1].groupby("protocol_id"):
        row[f"recall_{fam}"] = round(float(rej[g.index].mean()), 3)
    for s, g in calib[y == 1].groupby("source"):
        row[f"recall_src_{s}"] = round(float(rej[g.index].mean()), 3)
    out_csv = rep / "csv" / f"gate_calibration_frozen_{source}.csv"
    pd.DataFrame([row]).to_csv(out_csv, index=False)
    return out_csv, row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["webcode2m"], default="webcode2m")
    ap.add_argument("--target", type=float, default=0.99,
                    help="required composite breaking-recall")
    ap.add_argument("--slack", type=float, default=1.25,
                    help="multiplier on the safe-set max (safe_max policy)")
    ap.add_argument("--protocol-only", action="store_true",
                    help="exclude extension mutations (M8) from calibration")
    ap.add_argument("--freeze", action="store_true",
                    help="write config/gate_config.yaml")
    ap.add_argument("--force", action="store_true",
                    help="with --freeze, allow overwriting an existing config/gate_config.yaml")
    ap.add_argument("--evaluate-frozen", action="store_true",
                    help="load config/gate_config.yaml AS-IS and recompute recall/FPR on "
                         "the existing stress CSV with that exact policy; writes "
                         "gate_calibration_frozen_<src>.csv and changes nothing else")
    args = ap.parse_args()

    rep = ROOT / "reports"
    figs = rep / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    (rep / "csv").mkdir(parents=True, exist_ok=True)

    sources = [args.source]
    frames, csv_hashes = [], {}
    for s in sources:
        p = rep / "csv" / f"stress_samples_{s}.csv"
        if p.exists():
            frames.append(pd.read_csv(p))
            csv_hashes[s] = sha256(p)
        else:
            print(f"[04] missing {p} -- run step 03 for {s}")
    if not frames:
        sys.exit("[04] nothing to calibrate")
    full = pd.concat(frames, ignore_index=True)

    calib = full[(full["status"] == "ok")
                 & full["calib_include"].astype(bool)].copy()
    if args.protocol_only:
        calib = calib[~calib["extension"].astype(bool)]
    y = calib["verified_label"].astype(int).to_numpy()
    n_brk, n_safe = int(y.sum()), int((1 - y).sum())
    print(f"[04] calibration set: {len(calib)} samples "
          f"({n_brk} verified-breaking, {n_safe} verified-safe) "
          f"from {calib['page_id'].nunique()} pages / {sources}")
    if n_brk == 0 or n_safe == 0:
        sys.exit("[04] need both verified classes; run step 03 on more pages")

    cfg = load_config(ROOT / "config" / "gate_config.yaml")
    use_lpips = bool(cfg.get("use_lpips")) and \
        calib.get("lpips_max_tile") is not None and \
        pd.to_numeric(calib.get("lpips_max_tile"), errors="coerce").notna().any()

    if args.evaluate_frozen:
        out_csv, row = _evaluate_frozen(calib, y, cfg, use_lpips, n_brk,
                                        args.source, rep)
        print(f"[04] frozen policy on {args.source}: recall={row['recall']:.3f} "
              f"[{row['recall_ci_lo']:.3f}, {row['recall_ci_hi']:.3f}]  "
              f"fpr={row['fpr']:.3f}")
        print(f"[04] wrote {out_csv}")
        return

    S = metric_scores(calib)
    rows = []

    # ---------------- per-metric ROC / AUC / operating points -------------
    roc_store = {}
    for name, _fn, gate, _cfg_key, _to_cfg, floor in METRICS:
        v = S[name].to_numpy()
        result = roc(v, y)
        safe_scores = S[name][y == 0]
        brk_scores = S[name][y == 1]
        row = {"kind": "metric", "name": name, "gate": gate,
               "n_valid": int((~np.isnan(v)).sum()),
               "safe_max": float(np.nanmax(safe_scores)) if safe_scores.notna().any() else np.nan,
               "safe_p99": float(np.nanpercentile(safe_scores, 99)) if safe_scores.notna().any() else np.nan,
               "breaking_median": float(np.nanmedian(brk_scores)) if brk_scores.notna().any() else np.nan,
               "floor": floor}
        if result:
            fpr, tpr, thr, auc = result
            roc_store[name] = result
            row.update({"auc": round(auc, 4),
                        "recall_at_fpr0": round(max_recall_at_fpr(fpr, tpr, 0.0), 4),
                        "recall_at_fpr1pct": round(max_recall_at_fpr(fpr, tpr, 0.01), 4),
                        "recall_at_fpr5pct": round(max_recall_at_fpr(fpr, tpr, 0.05), 4)})
        rows.append(row)

    # structural detector (unmatched blocks) -- threshold-free
    unm = pd.to_numeric(calib["n_unmatched"], errors="coerce").fillna(0) > 0
    rows.append({"kind": "metric", "name": "n_unmatched>0", "gate": "G4",
                 "n_valid": len(calib),
                 "recall_at_fpr0": round(float(unm[y == 1].mean()), 4),
                 "auc": np.nan,
                 "safe_max": float(unm[y == 0].mean())})

    # SSIM diagnostics: per-sample BEST CASE of common-crop and union-canvas
    def _col(name):
        return (pd.to_numeric(calib[name], errors="coerce") if name in calib
                else pd.Series(np.nan, index=calib.index))
    ssim_crop = _col("ssim_diag")
    ssim_canv = _col("ssim_canvas")
    ssim_best = pd.concat([ssim_crop, ssim_canv], axis=1).max(axis=1)
    ssim_slips = {}
    for tau in SSIM_SLIP_THRESHOLDS:
        slip = float((ssim_best[y == 1] >= tau).mean())
        ssim_slips[tau] = slip
        rows.append({"kind": "ssim_diag", "name": f"ssim_best>= {tau}",
                     "gate": "SSIM",
                     "breaking_slip_rate": round(slip, 4),
                     "safe_accept_rate": round(float((ssim_best[y == 0] >= tau).mean()), 4)})
    ssim_roc = roc((1.0 - ssim_best).to_numpy(), y)
    if ssim_roc:
        roc_store["ssim_dissim(best)"] = ssim_roc
        rows.append({"kind": "ssim_diag", "name": "ssim_dissim(best)",
                     "gate": "SSIM", "auc": round(ssim_roc[3], 4),
                     "recall_at_fpr0": round(max_recall_at_fpr(ssim_roc[0], ssim_roc[1], 0.0), 4),
                     "recall_at_fpr5pct": round(max_recall_at_fpr(ssim_roc[0], ssim_roc[1], 0.05), 4)})

    # ---------------- composite policies ----------------------------------
    policies = {"provisional": thresholds_from_cfg(cfg)}
    safe_df = calib[y == 0]
    # safe_max asks "how loose can a threshold be and still clear every safe
    # sample?" -- but a pixel-identical sample is cleared by the soundness
    # rule regardless of its metrics, so letting it set the bound imports the
    # very DOM artifact the rule exists to neutralise (a 599px centre shift on
    # a byte-identical render). Only safe samples that COULD be rejected
    # constrain the threshold; if none remain, this data cannot bound the
    # metric from above and safe_max degenerates to the floor -- which is the
    # honest answer, not a tight gate earned by evidence.
    ident = pixel_identical_mask(calib).to_numpy()
    constraining = (y == 0) & ~ident
    n_constraining = int(constraining.sum())
    t_safe_max, t_strict = {}, {}
    for name, _fn, _g, _k, _t, floor in METRICS:
        col = S[name][constraining]
        smax = float(np.nanmax(col)) if col.notna().any() else 0.0
        t_safe_max[name] = max(floor, smax * args.slack)
        t_strict[name] = floor
    policies["safe_max"] = t_safe_max
    policies["strict"] = t_strict

    comp = {}
    for pname, raw_thr in policies.items():
        # Recall/FPR must be evaluated against the SAVED representation --
        # what config/gate_config.yaml will actually enforce -- not the raw
        # floats: int(ceil(...)) and round(..., N) can shift a threshold
        # enough to change which samples are (mis)classified. Round-tripping
        # through cfg keys applies the exact same to_cfg() transform used to
        # write the YAML, then maps back to metric-name keys.
        thr = thresholds_from_cfg(cfg_from_thresholds(raw_thr, cfg))
        rej = composite_reject(calib, thr, use_lpips)
        rec = float(rej[y == 1].mean())
        fpr_ = float(rej[y == 0].mean())
        lo, hi = wilson(int(rej[y == 1].sum()), n_brk)
        comp[pname] = {"thr": thr, "recall": rec, "fpr": fpr_,
                       "rej": rej, "recall_ci": (lo, hi)}
        row = {"kind": "composite", "name": pname,
               "recall": round(rec, 4), "recall_ci_lo": round(lo, 4),
               "recall_ci_hi": round(hi, 4), "fpr": round(fpr_, 4),
               "thresholds": json.dumps({k: round(float(v), 5)
                                         for k, v in thr.items()})}
        for fam, g in calib[y == 1].groupby("protocol_id"):
            row[f"recall_{fam}"] = round(float(rej[g.index].mean()), 3)
        for src, g in calib[y == 1].groupby("source"):
            row[f"recall_src_{src}"] = round(float(rej[g.index].mean()), 3)
        comp[pname]["row"] = row
        rows.append(row)

    # frozen pick: loosest qualifying policy, else best recall (warned)
    order = ["safe_max", "strict"]
    qualifying = [p for p in order if comp[p]["recall"] >= args.target]
    frozen = qualifying[0] if qualifying else max(order, key=lambda p: comp[p]["recall"])
    frozen_thr = comp[frozen]["thr"]
    frozen_rej = comp[frozen]["rej"]

    # per-mutation detection at frozen thresholds (+ severity views)
    det_rows = []
    for (pidid, var), g in calib[y == 1].groupby(["protocol_id", "variant"]):
        k = int(frozen_rej[g.index].sum())
        lo, hi = wilson(k, len(g))
        det_rows.append({"kind": "detection", "name": f"{pidid}:{var}",
                         "n": len(g), "detected": k,
                         "rate": round(k / len(g), 4),
                         "ci_lo": round(lo, 4), "ci_hi": round(hi, 4),
                         "ssim_slip_0.99": round(float(
                             (ssim_best[g.index] >= 0.99).mean()), 4)})
    rows.extend(det_rows)

    # ---------------- decision rules (pre-registered) ----------------------
    # R1: SSIM kill-shot. Best-case SSIM (max of crop/canvas variants) at
    # conventional thresholds vs the composite gate at frozen thresholds.
    r1_slip_099 = ssim_slips.get(0.99, float("nan"))
    r1_slip_095 = ssim_slips.get(0.95, float("nan"))
    composite_recall = comp[frozen]["recall"]
    r1 = {"ssim_slip_at_0.99": r1_slip_099, "ssim_slip_at_0.95": r1_slip_095,
          "composite_recall_frozen": composite_recall,
          "verdict": ("SSIM alone is insufficient: at the conventional 0.99 "
                      f"threshold it waves through {100*r1_slip_099:.1f}% of "
                      "verified-breaking mutants that the layered gate "
                      f"rejects at {100*composite_recall:.1f}% recall."
                      if r1_slip_099 > 0.05 else
                      "SSIM slip rate at 0.99 was below 5%; the kill-shot "
                      "claim needs the per-severity table, not the headline.")}

    # R2: LPIPS marginal value beyond the hard gates at frozen thresholds.
    r2 = {"evaluated": use_lpips}
    if use_lpips:
        hard_thr = dict(frozen_thr)
        hard_rej = composite_reject(calib, hard_thr, use_lpips=False)
        lp = S["lpips_max_tile"] > frozen_thr["lpips_max_tile"]
        marginal = int((lp & ~hard_rej & (pd.Series(y, index=calib.index) == 1)).sum())
        r2.update({"marginal_detections": marginal,
                   "verdict": ("LPIPS adds no detections beyond G2-G5; drop "
                               "G6 in production and say so." if marginal == 0
                               else f"LPIPS adds {marginal} detections the "
                                    "hard gates miss; keep G6.")})
    else:
        r2["verdict"] = ("LPIPS column absent or use_lpips=false; rule "
                         "deferred (run step 03 with use_lpips: true to "
                         "evaluate G6).")

    # findings outside the calibration set
    finds = full[full["status"] == "unsafe_safe"]
    duds = full[full["status"] == "dud"]
    probe = full[(full["intent"] == "probe") & (full["status"] == "ok")]

    # ---------------- outputs ---------------------------------------------
    calib_csv = rep / "csv" / "gate_calibration.csv"
    pd.DataFrame(rows).to_csv(calib_csv, index=False)

    prov = {"frozen_policy": frozen,
            "frozen_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "git_commit": git_commit(),
            "calibration_sources": sources,
            "stress_csv_sha256": csv_hashes,
            "n_breaking": n_brk, "n_safe": n_safe,
            "n_safe_constraining": n_constraining,
            "target_recall": args.target, "slack": args.slack,
            "composite_recall": round(composite_recall, 4),
            "composite_fpr": round(comp[frozen]["fpr"], 4)}
    new_cfg = cfg_from_thresholds(frozen_thr, cfg)
    if use_lpips and r2.get("marginal_detections") == 0:
        new_cfg["use_lpips"] = False
    _write_yaml(ROOT / "config" / "gate_config.calibrated.yaml", new_cfg, prov)
    freeze_refused = None
    if args.freeze:
        gate_yaml = ROOT / "config" / "gate_config.yaml"
        if not qualifying:
            freeze_refused = (f"no policy reached the target recall "
                              f"{args.target:.0%} (best: {frozen} at "
                              f"{composite_recall:.1%})")
        elif gate_yaml.exists() and not args.force:
            freeze_refused = f"{gate_yaml} already exists; pass --force to overwrite it"
        if freeze_refused:
            print(f"[04] REFUSING to freeze: {freeze_refused}")
        else:
            _write_yaml(gate_yaml, new_cfg, prov)
            print(f"[04] FROZE {gate_yaml} -- commit it now:")
            print('     git add config/gate_config.yaml reports/ && '
                  'git commit -m "FREEZE gate thresholds after stress test"')

    _decisions_md(rep / "stress_decisions.md", args, comp, frozen, qualifying,
                  det_rows, r1, r2, finds, duds, probe, prov, calib, y,
                  frozen_rej)

    # Figures last and non-fatally: they are a deliverable, but an optional
    # plotting dependency must never cost you the frozen config or the
    # decisions log (both already written above).
    fig_err = None
    try:
        _figures(figs, roc_store, det_rows, calib, y, frozen_rej, ssim_best)
    except ImportError as e:  # noqa: BLE001
        fig_err = f"{e} -- run: pip install matplotlib, then re-run step 04"
    except Exception as e:  # noqa: BLE001
        fig_err = repr(e)

    print(f"[04] wrote {calib_csv}")
    if fig_err:
        print(f"[04] FIGURES SKIPPED: {fig_err}")
    else:
        print(f"[04] wrote {figs}/roc_grid.png, detection_heatmap.png, "
              f"severity_curves.png")
    print(f"[04] wrote config/gate_config.calibrated.yaml "
          f"(policy={frozen}, recall={composite_recall:.3f}, "
          f"fpr={comp[frozen]['fpr']:.3f})")
    print(f"[04] wrote reports/stress_decisions.md")
    if not qualifying:
        miss = [d for d in det_rows if d["rate"] < 1.0]
        print(f"[04] WARNING: no policy reached recall >= {args.target}. "
              f"Slipping classes: "
              + ", ".join(f"{d['name']} ({d['rate']:.0%})" for d in miss))
    if freeze_refused:
        sys.exit(f"[04] --freeze refused: {freeze_refused}")


# ---------------------------------------------------------------------------
def _figures(figs, roc_store, det_rows, calib, y, frozen_rej, ssim_best):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # ROC grid
    names = list(roc_store)
    if names:
        n = len(names)
        cols = 4
        rws = int(np.ceil(n / cols))
        fig, axes = plt.subplots(rws, cols, figsize=(3.2 * cols, 3.0 * rws))
        axes = np.atleast_1d(axes).ravel()
        for ax, name in zip(axes, names):
            fpr, tpr, _thr, auc = roc_store[name]
            ax.plot(fpr, tpr, lw=1.8)
            ax.plot([0, 1], [0, 1], "k:", lw=0.8)
            ax.set_title(f"{name}\nAUC={auc:.3f}", fontsize=9)
            ax.set_xlabel("safe false-rejection", fontsize=8)
            ax.set_ylabel("breaking recall", fontsize=8)
            ax.tick_params(labelsize=7)
        for ax in axes[len(names):]:
            ax.axis("off")
        fig.suptitle("Per-metric ROC on verified stress labels", fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        fig.savefig(figs / "roc_grid.png", dpi=160)
        plt.close(fig)

    # detection heatmap: mutation class x (per-metric standalone + composite)
    brk = calib[y == 1]
    if len(brk):
        S = metric_scores(calib)
        classes = sorted(brk.groupby(["protocol_id", "variant"]).groups)
        col_names = [m[0] for m in METRICS if S[m[0]].notna().any()] + \
                    ["unmatched", "ssim<0.99", "COMPOSITE"]
        # standalone per-metric detection uses the frozen thresholds implied
        # by det_rows' composite; approximate standalone via score > safe_max
        # of that metric on the calibration set (zero-FPR operating point).
        grid = np.zeros((len(classes), len(col_names)))
        for i, key in enumerate(classes):
            g = brk[(brk["protocol_id"] == key[0]) & (brk["variant"] == key[1])]
            for j, cn in enumerate(col_names):
                if cn == "COMPOSITE":
                    grid[i, j] = frozen_rej[g.index].mean()
                elif cn == "unmatched":
                    grid[i, j] = (pd.to_numeric(g["n_unmatched"],
                                                errors="coerce").fillna(0) > 0).mean()
                elif cn == "ssim<0.99":
                    grid[i, j] = (ssim_best[g.index] < 0.99).mean()
                else:
                    v = S[cn]
                    safe_max = float(np.nanmax(v[y == 0])) if v[y == 0].notna().any() else 0.0
                    grid[i, j] = (v[g.index] > safe_max).mean()
        fig, ax = plt.subplots(figsize=(1.1 * len(col_names) + 2,
                                        0.45 * len(classes) + 2))
        im = ax.imshow(grid, vmin=0, vmax=1, cmap="RdYlGn", aspect="auto")
        ax.set_xticks(range(len(col_names)))
        ax.set_xticklabels(col_names, rotation=40, ha="right", fontsize=8)
        ax.set_yticks(range(len(classes)))
        ax.set_yticklabels([f"{p}:{v}" for p, v in classes], fontsize=8)
        for i in range(len(classes)):
            for j in range(len(col_names)):
                ax.text(j, i, f"{grid[i, j]:.2f}", ha="center", va="center",
                        fontsize=7,
                        color="black" if 0.25 < grid[i, j] < 0.8 else "white")
        ax.set_title("Detection rate per verified-breaking class\n"
                     "(per-metric at its zero-FPR point; COMPOSITE at frozen "
                     "thresholds)", fontsize=10)
        fig.colorbar(im, ax=ax, shrink=0.8)
        fig.tight_layout()
        fig.savefig(figs / "detection_heatmap.png", dpi=160)
        plt.close(fig)

    # severity curves for the graded mutations (M3 px, M4 dE00, M5 +/-px)
    graded = [("M3", "margin-left shift (px)"),
              ("M4", "background dE00 (achieved)"),
              ("M5", "font-size delta (px)")]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
    plotted = False
    for ax, (pidid, xlabel) in zip(axes, graded):
        g = brk[brk["protocol_id"] == pidid]
        if not len(g):
            ax.axis("off")
            continue
        sev = pd.to_numeric(
            g["severity_achieved"].where(g["severity_achieved"].notna(),
                                         g["severity_nominal"]),
            errors="coerce")
        buckets = pd.to_numeric(g["severity_nominal"], errors="coerce")
        xs, comp_y, ssim_y, ns = [], [], [], []
        for b in sorted(buckets.dropna().unique()):
            m = buckets == b
            xs.append(float(np.nanmean(sev[m])) if sev[m].notna().any() else b)
            comp_y.append(frozen_rej[g.index[m]].mean())
            ssim_y.append((ssim_best[g.index[m]] < 0.99).mean())
            ns.append(int(m.sum()))
        ax.plot(xs, comp_y, "o-", label="layered gate (frozen)")
        ax.plot(xs, ssim_y, "s--", label="SSIM @ 0.99")
        for x_, n_ in zip(xs, ns):
            ax.annotate(f"n={n_}", (x_, 1.02), fontsize=7, ha="center")
        ax.set_ylim(-0.05, 1.12)
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_ylabel("detection rate", fontsize=9)
        ax.set_title(pidid, fontsize=10)
        ax.legend(fontsize=7)
        plotted = True
    if plotted:
        fig.suptitle("Detection vs mutation severity -- where thresholds "
                     "come from, and where SSIM goes blind", fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        fig.savefig(figs / "severity_curves.png", dpi=160)
    plt.close(fig)


def _decisions_md(path, args, comp, frozen, qualifying, det_rows, r1, r2,
                  finds, duds, probe, prov, calib, y, frozen_rej):
    L = []
    L.append("# Gate calibration decisions (auto-generated by step 04)\n")
    L.append(f"_frozen policy: **{frozen}**; generated "
             f"{prov['frozen_at_utc']}; git {prov['git_commit']}; "
             f"calibration set: {prov['n_breaking']} verified-breaking / "
             f"{prov['n_safe']} verified-safe from "
             f"{', '.join(prov['calibration_sources'])}_\n")
    L.append("## Composite gate (visual gates G2-G6, verified labels)\n")
    L.append("| policy | breaking recall | 95% CI | safe false-rejection |")
    L.append("|---|---|---|---|")
    for p, c in comp.items():
        lo, hi = c["recall_ci"]
        mark = " **<- frozen**" if p == frozen else ""
        L.append(f"| {p}{mark} | {c['recall']:.3f} | [{lo:.3f}, {hi:.3f}] "
                 f"| {c['fpr']:.3f} |")
    n_con = prov.get("n_safe_constraining", prov["n_safe"])
    n_ident = prov["n_safe"] - n_con
    if n_con == 0:
        L.append(f"\n> **False-rejection is not measured on this set.** All "
                 f"{prov['n_safe']} verified-safe samples render pixel-"
                 f"identically and are accepted by the soundness rule "
                 f"regardless of thresholds, so none could be rejected at all. "
                 f"A false-rejection of 0.000 is true by construction, not "
                 f"evidence that the thresholds are well placed, and "
                 f"`safe_max` degenerates to the floors because nothing bounds "
                 f"the metrics from above. Add safe mutants that change pixels "
                 f"below the perceptual JND (e.g. a dE00~0.5 recolour) to give "
                 f"this axis information.")
    elif n_ident:
        L.append(f"\n> **Basis of the false-rejection column.** {n_ident} of "
                 f"{prov['n_safe']} verified-safe samples render pixel-"
                 f"identically and are accepted by the soundness rule at any "
                 f"threshold, so they cannot contribute false rejections. "
                 f"Every rejection in the column above therefore comes from "
                 f"the {n_con} sub-JND samples that DO change pixels, but the "
                 f"rate is reported over all {prov['n_safe']} negatives, so it "
                 f"is diluted by a factor of about "
                 f"{prov['n_safe'] / max(n_con, 1):.1f}. Read it against the "
                 f"constraining class instead: see the sub-JND acceptance rate "
                 f"below, which uses {n_con} as its denominator. Quote "
                 f"whichever you mean, and say which.")
    if "intent" in calib.columns:
        tmask = (calib["intent"] == "tolerable").to_numpy()
        n_tol = int(tmask.sum())
        if n_tol:
            rej = np.asarray(frozen_rej)[tmask]
            acc = 1.0 - float(rej.mean())
            L.append("\n### Sub-JND negatives (the constraining class)\n")
            L.append(f"{n_tol} pixel-DIFFERENT but imperceptible mutants; "
                     f"accepted at frozen thresholds: **{acc:.3f}**. Pixel-"
                     f"identical negatives are accepted by the soundness rule "
                     f"at any threshold, so these are the only samples able to "
                     f"reject -- they alone bound every threshold from above.")
            if "severity_achieved" in calib.columns:
                sev = pd.to_numeric(calib.loc[tmask, "severity_achieved"],
                                    errors="coerce")
                if sev.notna().any():
                    L.append(f"\nMax achieved dE00 = {sev.max():.2f} "
                             f"(JND ~= 1.0), recorded per sample, so the "
                             f"imperceptibility claim is bounded by "
                             f"construction and auditable rather than "
                             f"asserted.")
    if not qualifying:
        L.append(f"\n**No policy reached the {args.target:.0%} recall "
                 f"target.** Slipping classes (frozen thresholds):")
        for d in det_rows:
            if d["rate"] < 1.0:
                L.append(f"- {d['name']}: {d['rate']:.0%} "
                         f"[{d['ci_lo']:.2f}, {d['ci_hi']:.2f}] (n={d['n']})")
    L.append("\n## Per-mutation detection at frozen thresholds\n")
    L.append("| class | n | detected | rate | 95% CI | SSIM>=0.99 slip |")
    L.append("|---|---|---|---|---|---|")
    for d in sorted(det_rows, key=lambda d: d["name"]):
        L.append(f"| {d['name']} | {d['n']} | {d['detected']} | "
                 f"{d['rate']:.2f} | [{d['ci_lo']:.2f}, {d['ci_hi']:.2f}] | "
                 f"{d['ssim_slip_0.99']:.2f} |")
    L.append("\n## R1 -- pre-registered: is SSIM alone sufficient?\n")
    L.append(f"- best-case SSIM (max of common-crop and union-canvas) slip "
             f"on verified-breaking: {r1['ssim_slip_at_0.95']:.1%} at 0.95, "
             f"{r1['ssim_slip_at_0.99']:.1%} at 0.99")
    L.append(f"- layered gate recall at frozen thresholds: "
             f"{r1['composite_recall_frozen']:.1%}")
    L.append(f"- **verdict:** {r1['verdict']}")
    L.append("\n## R2 -- pre-registered: does LPIPS (G6) add value?\n")
    L.append(f"- **verdict:** {r2['verdict']}")
    L.append("\n## Findings outside the calibration set\n")
    L.append(f"- duds (breaking intent, zero pixel change after resampling): "
             f"{len(duds)}"
             + (f" -- by op: {duds.groupby('mutation').size().to_dict()}"
                if len(duds) else ""))
    L.append(f"- unsafe-safe (safe intent, pixels CHANGED -- 'safety is not "
             f"decidable statically', measured): {len(finds)}"
             + (f" -- by op: {finds.groupby('mutation').size().to_dict()}"
                if len(finds) else ""))
    if len(probe):
        L.append(f"- X1 legacy whitespace regex broke rendering on "
                 f"{probe['verified_change'].mean():.1%} of pages "
                 f"({int(probe['verified_change'].sum())}/{len(probe)}) -- "
                 f"the measured case for spec-aware minify-html")
    L.append("\n_Thresholds and provenance: config/gate_config.calibrated.yaml"
             + (" (frozen into gate_config.yaml)" if args.freeze else
                "; re-run with --freeze to promote") + "._\n")
    Path(path).write_text("\n".join(L), encoding="utf-8")


def _write_yaml(path: Path, cfg: dict, prov: dict) -> None:
    import yaml
    hdr = ["# Acceptance Gate v2 thresholds -- CALIBRATED by the mutation "
           "stress test (steps 03+04).",
           "# Do not edit by hand after the freeze commit."]
    hdr += [f"# {k}: {v}" for k, v in prov.items()]
    body = yaml.safe_dump(cfg, sort_keys=False)
    Path(path).write_text("\n".join(hdr) + "\n" + body, encoding="utf-8")


if __name__ == "__main__":
    main()