"""Step 02 -- run upgraded Level 1 over the pilot manifest and evaluate every
page through Acceptance Gate v2. This regenerates your Level-1 table under a
sound gate (your old numbers were produced under the broken one).

Usage:
    python src/pipeline/02_run_level1.py --source webcode2m

Outputs:
    outputs/level1/<source>/<id>.html          (minified pages)
    outputs/renders_gate/<source>/l1/*.png     (gate screenshots)
    reports/csv/level1_gate_<source>.csv       (per-page gate results)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from tqdm import tqdm  # noqa: E402

from src.compare.gate import load_config, run_gate  # noqa: E402
from src.compress.level1_minify import level1_minify  # noqa: E402
from src.render.harness import RenderHarness  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["webcode2m"], default="webcode2m")
    args = ap.parse_args()

    manifest = ROOT / "data" / "splits" / f"pilot_{args.source}_manifest.csv"
    df = pd.read_csv(manifest)

    l1_dir = ROOT / "outputs" / "level1" / args.source
    l1_dir.mkdir(parents=True, exist_ok=True)
    work = ROOT / "outputs" / "renders_gate" / args.source / "l1"
    rep = ROOT / "reports" / "csv"
    rep.mkdir(parents=True, exist_ok=True)

    cfg = load_config(ROOT / "config" / "gate_config.yaml")
    rows = []
    with RenderHarness() as h:
        for _, r in tqdm(df.iterrows(), total=len(df), desc="level1+gate"):
            pid = r["page_id"]
            try:
                html = Path(r["html_path"]).read_text(encoding="utf-8",
                                                      errors="ignore")
                mini = level1_minify(html)
                out_html = l1_dir / f"{pid}.html"
                out_html.write_text(mini, encoding="utf-8")
                res = run_gate(r["html_path"], out_html, h, work, pid, cfg)
                res["status"] = "ok"
            except Exception as e:  # noqa: BLE001
                res = {"page_id": pid, "accepted": False, "status": f"failed: {e}"}
            rows.append(res)

    out = pd.DataFrame(rows)
    out_csv = rep / f"level1_gate_{args.source}.csv"
    out.to_csv(out_csv, index=False)

    ok = out[out.get("status", "ok") == "ok"]
    if len(ok):
        red = ok["reduction_pct"].astype(float)
        print("\n[02] ---- Level 1 under Gate v2 ----")
        print(f"accepted: {int(ok['accepted'].sum())}/{len(out)}")
        print(f"reduction  mean {red.mean():.2f}%   median {np.median(red):.2f}%   "
              f"p10 {np.percentile(red, 10):.2f}%   p90 {np.percentile(red, 90):.2f}%")
        if "ssim_diag" in ok:
            print(f"SSIM (diagnostic only) mean: {ok['ssim_diag'].astype(float).mean():.5f}")
        rej = ok[~ok["accepted"].astype(bool)]
        if len(rej):
            gates = [c for c in ["g0_parse", "g1_tokens", "g2_height", "g3_text",
                                 "g4_blocks", "g5_color"] if c in rej]
            print("\nrejected pages and which gate tripped:")
            print(rej[["page_id"] + gates].to_string(index=False))
    print(f"[02] wrote {out_csv}")
    print("[02] NEXT: python src/pipeline/03_run_stress_test.py --source", args.source)


if __name__ == "__main__":
    main()
