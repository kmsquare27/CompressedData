"""Step 01 -- determinism audit. Render every pilot page TWICE under the
harness and require pixel-identical output. Until this passes, no metric in
the project means anything.

Usage:
    python src/pipeline/01_determinism_audit.py --source webcode2m

Definition of Done: >= 99% identical. Investigate every flake (usually an
animation that escaped the freeze, or an asset that wasn't intercepted).
This audit becomes one sentence in Threats to Validity: "renderer
nondeterminism was measured, not assumed."
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402
from tqdm import tqdm  # noqa: E402

from src.render.harness import RenderHarness  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["webcode2m"], default="webcode2m")
    args = ap.parse_args()

    manifest = ROOT / "data" / "splits" / f"pilot_{args.source}_manifest.csv"
    df = pd.read_csv(manifest)
    out_dir = ROOT / "outputs" / "audit" / args.source
    rep_dir = ROOT / "reports" / "csv"
    rep_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    with RenderHarness() as h:
        for _, r in tqdm(df.iterrows(), total=len(df), desc="determinism audit"):
            pid = r["page_id"]
            a = out_dir / f"{pid}_a.png"
            b = out_dir / f"{pid}_b.png"
            try:
                h.render(r["html_path"], a)
                h.render(r["html_path"], b)
                A = np.asarray(Image.open(a))
                B = np.asarray(Image.open(b))
                same = A.shape == B.shape and bool(np.array_equal(A, B))
                diff = 0 if same else int(np.sum(np.any(A[:min(len(A), len(B))]
                                                        != B[:min(len(A), len(B))],
                                                        axis=-1)))
                rows.append({"page_id": pid, "identical": same,
                             "shape_a": str(A.shape), "shape_b": str(B.shape),
                             "diff_pixels": diff, "status": "ok"})
            except Exception as e:  # noqa: BLE001
                rows.append({"page_id": pid, "identical": False,
                             "shape_a": "", "shape_b": "", "diff_pixels": -1,
                             "status": f"failed: {e}"})

    out = pd.DataFrame(rows)
    out_csv = rep_dir / f"determinism_audit_{args.source}.csv"
    out.to_csv(out_csv, index=False)

    n_ok = int(out["identical"].sum())
    print(f"\n[01] identical: {n_ok}/{len(out)}  ({100.0 * n_ok / max(len(out), 1):.1f}%)")
    flaky = out[~out["identical"]]
    if len(flaky):
        print("[01] flaky pages (open their two renders side by side):")
        print(flaky[["page_id", "status", "diff_pixels"]].to_string(index=False))
    print(f"[01] wrote {out_csv}")
    print("[01] NEXT: python src/pipeline/02_run_level1.py --source", args.source)


if __name__ == "__main__":
    main()
