"""Step 03 -- the gate stress test (metric validation).

Applies every mutation to K pilot pages, runs the gate on each mutant, and
reports per-mutation detection: a sound gate REJECTS breaking mutants and
ACCEPTS safe ones. The CSV also stores SSIM per mutant, so you can show
exactly how many breakages SSIM would wave through at any threshold -- your
quantitative case that SSIM alone is insufficient, and the calibration data
for freezing config/gate_config.yaml.

Usage:
    python src/pipeline/03_run_stress_test.py --source webcode2m --k 20
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402
from tqdm import tqdm  # noqa: E402

from src.compare.gate import load_config, run_gate  # noqa: E402
from src.render.harness import RenderHarness  # noqa: E402
from src.stress.mutations import MUTATIONS, apply_mutation  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["webcode2m"], default="webcode2m")
    ap.add_argument("--k", type=int, default=20, help="pages per mutation")
    args = ap.parse_args()

    df = pd.read_csv(ROOT / "data" / "splits" /
                     f"pilot_{args.source}_manifest.csv").head(args.k)
    mut_dir = ROOT / "outputs" / "stress" / args.source
    mut_dir.mkdir(parents=True, exist_ok=True)
    work = ROOT / "outputs" / "renders_gate" / args.source / "stress"
    rep = ROOT / "reports" / "csv"
    rep.mkdir(parents=True, exist_ok=True)
    cfg = load_config(ROOT / "config" / "gate_config.yaml")

    rows = []
    with RenderHarness() as h:
        for _, r in df.iterrows():
            pid = r["page_id"]
            html = Path(r["html_path"]).read_text(encoding="utf-8", errors="ignore")
            for mname in tqdm(MUTATIONS, desc=f"{pid}", leave=False):
                mutated, label = apply_mutation(mname, html, seed=hash((pid, mname)) & 0xFFFF)
                if mutated is None:
                    continue
                mpath = mut_dir / f"{pid}__{mname}.html"
                mpath.write_text(mutated, encoding="utf-8")
                try:
                    res = run_gate(r["html_path"], mpath, h, work,
                                   f"{pid}__{mname}", cfg)
                    res.update({"mutation": mname, "label_breaking": label,
                                "status": "ok"})
                except Exception as e:  # noqa: BLE001
                    res = {"page_id": pid, "mutation": mname,
                           "label_breaking": label, "accepted": False,
                           "status": f"failed: {e}"}
                rows.append(res)

    out = pd.DataFrame(rows)
    out_csv = rep / f"stress_test_{args.source}.csv"
    out.to_csv(out_csv, index=False)

    ok = out[out["status"] == "ok"].copy()
    ok["accepted"] = ok["accepted"].astype(bool)
    print("\n[03] ---- Gate stress test ----")
    brk = ok[ok["label_breaking"] == 1]
    safe = ok[ok["label_breaking"] == 0]
    if len(brk):
        det = 100.0 * (~brk["accepted"]).mean()
        print(f"BREAKING mutants rejected by gate: {det:.1f}%  (target >= 99%)")
        print(brk.groupby("mutation")["accepted"]
                 .apply(lambda s: f"{100.0 * (~s).mean():.0f}% rejected").to_string())
    if len(safe):
        acc = 100.0 * safe["accepted"].mean()
        print(f"\nSAFE mutants accepted by gate:     {acc:.1f}%  (target: high)")
        print(safe.groupby("mutation")["accepted"]
                  .apply(lambda s: f"{100.0 * s.mean():.0f}% accepted").to_string())
    if "ssim_diag" in ok and len(brk):
        high_ssim_breaks = (brk["ssim_diag"].astype(float) >= 0.95).mean() * 100.0
        print(f"\nBreaking mutants with SSIM >= 0.95 (would slip an SSIM-only gate): "
              f"{high_ssim_breaks:.1f}%   <-- your headline number")
    print(f"[03] wrote {out_csv}")
    print("[03] NEXT: inspect per-gate columns, tune config/gate_config.yaml, "
          "re-run, then FREEZE thresholds and move to Level 2 (protocol Stage 4).")


if __name__ == "__main__":
    main()
