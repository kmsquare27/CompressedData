"""Step 02 (v3) -- Level 1 in two gated stages.

    L1a  render-neutral by construction (CSS-only minification, pruning)
    L1b  L1a + minify-html HTML pass (whitespace, quotes) -- expected to be
         rejected on pages where minify-html deletes whitespace between
         inline children of layout tags (see compress/level1_minify.py)

Both stages are gated against ONE cached render of the ORIGINAL, never
against each other, so nothing accumulates. The page's Level-1 artifact is
the fewest-token stage the gate accepted; that file goes to
outputs/level1/<source>/<id>.html (what step 06 composes on) and its row to
reports/csv/level1_gate_<source>.csv (what steps 06/07 read: page_id,
accepted, tokens_orig, tokens_comp, ...). Every stage's row is kept in
reports/csv/level1_stages_<source>.csv for the paper's per-stage table.

Usage:
    python src/pipeline/02_run_level1.py --source webcode2m
    python src/pipeline/02_run_level1.py --source webcode2m --no-html-pass
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from tqdm import tqdm  # noqa: E402

from src.compare.gate import (TokenCounter, evaluate_pair, load_config,  # noqa: E402
                              render_page)
from src.compress.level1_minify import level1a, level1b  # noqa: E402
from src.render.harness import RenderHarness  # noqa: E402


def _check_minify_html():
    import inspect
    import minify_html
    params = inspect.signature(minify_html.minify).parameters
    if "minify_doctype" not in params:
        raise SystemExit("[02] minify-html < 0.16 detected (no `minify_doctype`). "
                         "Pin minify-html>=0.16; the old `do_not_minify_doctype` "
                         "call fell back silently to defaults.")


def summarize(stages_df: pd.DataFrame, out: pd.DataFrame, out_csv: Path, source: str) -> None:
    print("\n[02] ---- Level 1 under the frozen gate, per stage ----")
    for stage in ("l1a", "l1b"):
        s = stages_df[(stages_df.get("stage") == stage) & (stages_df.get("status") == "ok")]
        if not len(s):
            continue
        acc = s["accepted"].astype(bool)
        red = pd.to_numeric(s.loc[acc, "reduction_pct"], errors="coerce")
        pix = s["pixel_identical"].astype(bool).mean() if "pixel_identical" in s else float("nan")
        print(f"{stage}: accepted {int(acc.sum())}/{len(s)}  pixel-identical "
              f"{100 * pix:.0f}%  reduction on accepted mean "
              f"{red.mean():.2f}% median {red.median():.2f}% p90 {red.quantile(.9):.2f}%")
        rej = s[~acc]
        if len(rej) and "center_shift_max" in rej:
            cs = pd.to_numeric(rej["center_shift_max"], errors="coerce")
            print(f"      rejected {len(rej)}: median centre shift {cs.median():.1f}px "
                  f"(~4px = one collapsed inter-inline space), "
                  f"G4 tripped on {int((~rej['g4_blocks'].astype(bool)).sum())}")
    sel = out[out.get("status") == "ok"]
    if len(sel):
        acc = sel["accepted"].astype(bool)
        red = pd.to_numeric(sel.loc[acc, "reduction_pct"], errors="coerce")
        print(f"\nselected: accepted {int(acc.sum())}/{len(out)}  "
              f"(l1a {int((sel.loc[acc, 'stage'] == 'l1a').sum())}, "
              f"l1b {int((sel.loc[acc, 'stage'] == 'l1b').sum())})  "
              f"mean {red.mean():.2f}%  page-average reduction "
              f"{red.sum() / max(len(out), 1):.2f}%")
        to = pd.to_numeric(out["tokens_orig"], errors="coerce")
        tc = pd.to_numeric(out["tokens_comp"], errors="coerce").fillna(to)
        print(f"corpus-level token saving (rejected pages count as 0%): "
              f"{100.0 * (to.sum() - tc.sum()) / max(to.sum(), 1):.2f}%")
    tok = (out["tokenizer"].dropna().iloc[0]
           if "tokenizer" in out and out["tokenizer"].notna().any() else "?")
    print(f"[02] tokenizer: {tok}")
    print(f"[02] wrote {out_csv}")
    print("[02] NEXT: python src/pipeline/03_run_stress_test.py --source", source)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["webcode2m"], default="webcode2m")
    ap.add_argument("--no-html-pass", dest="html_pass", action="store_false",
                    default=True, help="skip L1b (HTML-syntax pass)")
    ap.add_argument("--report-only", action="store_true",
                    help="skip rendering; re-print the summary from the existing gate CSVs")
    args = ap.parse_args()

    rep = ROOT / "reports" / "csv"
    out_csv = rep / f"level1_gate_{args.source}.csv"
    stages_csv = rep / f"level1_stages_{args.source}.csv"
    if args.report_only:
        if not out_csv.exists() or not stages_csv.exists():
            raise SystemExit(f"[02] --report-only needs {out_csv.name} and "
                             f"{stages_csv.name}; run without it first")
        summarize(pd.read_csv(stages_csv), pd.read_csv(out_csv), out_csv, args.source)
        return
    _check_minify_html()

    manifest = ROOT / "data" / "splits" / f"pilot_{args.source}_manifest.csv"
    df = pd.read_csv(manifest)

    dirs = {s: ROOT / "outputs" / f"level1{s[-1]}" / args.source for s in ("l1a", "l1b")}
    final_dir = ROOT / "outputs" / "level1" / args.source
    for d in [*dirs.values(), final_dir]:
        d.mkdir(parents=True, exist_ok=True)
    work = ROOT / "outputs" / "renders_gate" / args.source / "l1"
    work.mkdir(parents=True, exist_ok=True)
    rep.mkdir(parents=True, exist_ok=True)

    cfg = load_config(ROOT / "config" / "gate_config.yaml")
    tk = TokenCounter.get()
    stage_rows, page_rows = [], []

    with RenderHarness() as h:
        for _, r in tqdm(df.iterrows(), total=len(df), desc="level1[a,b]+gate"):
            pid = str(r["page_id"])
            try:
                html = Path(r["html_path"]).read_text(encoding="utf-8", errors="ignore")
                art_o = render_page(h, r["html_path"], work / f"{pid}_orig.png", tk)
            except Exception as e:  # noqa: BLE001
                page_rows.append({"page_id": pid, "stage": "none", "accepted": False,
                                  "status": f"failed_original: {e}"})
                continue

            results = []                    # (stage, R, out_path)
            stages = [("l1a", level1a, html)]
            for stage, fn, src in stages:
                try:
                    out_html = fn(src)
                    p = dirs[stage] / f"{pid}.html"
                    p.write_text(out_html, encoding="utf-8")
                    art_c = render_page(h, p, work / f"{pid}_{stage}.png", tk)
                    R = evaluate_pair(art_o, art_c, cfg, page_id=pid)
                    R.update({"stage": stage, "status": "ok"})
                    results.append((stage, R, p))
                    if stage == "l1a" and args.html_pass:
                        stages.append(("l1b", level1b, out_html))   # L1b runs on L1a
                except Exception as e:  # noqa: BLE001
                    results.append((stage, {"page_id": pid, "stage": stage,
                                            "accepted": False,
                                            "status": f"failed: {e}"}, None))
            for stage, R, _ in results:
                stage_rows.append(R)

            ok = [(s, R, p) for s, R, p in results
                  if p is not None and bool(R.get("accepted"))]
            if ok:
                stage, R, p = min(ok, key=lambda t: t[1]["tokens_comp"])
                shutil.copyfile(p, final_dir / f"{pid}.html")
                page_rows.append(dict(R))
            else:
                # Nothing accepted: record the L1a row (accepted False) and
                # write NO outputs/level1 file, so step 06 restarts from the
                # original for this page.
                (final_dir / f"{pid}.html").unlink(missing_ok=True)
                page_rows.append(dict(results[0][1]))

    stages_df = pd.DataFrame(stage_rows)
    stages_df.to_csv(rep / f"level1_stages_{args.source}.csv", index=False)
    out = pd.DataFrame(page_rows)
    out.to_csv(out_csv, index=False)

    summarize(stages_df, out, out_csv, args.source)


if __name__ == "__main__":
    main()
