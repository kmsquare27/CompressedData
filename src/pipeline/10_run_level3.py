"""Step 10 -- Level 3: invisible CSS under the training render, gate-validated.

Composition: runs on each page's SELECTED artifact from step 07 (the fewest-
token gate-accepted output of Levels 1-2), because Level 2's removals make
more rules dead. Acceptance is against the render of the ORIGINAL file and
requires pixel identity (--strict, default) in addition to the frozen gate
and G1; the tolerance gate alone is blind to style-detail damage (see
compress/level3_css.py).

Usage:
    python src/pipeline/10_run_level3.py --source webcode2m
    python src/pipeline/10_run_level3.py --source webcode2m --ablate-declarations
    python src/pipeline/10_run_level3.py --source webcode2m --merge-identical
    python src/pipeline/10_run_level3.py --source webcode2m --on-original

Output: outputs/level3/<source>/<id>.html          accepted pages ONLY (this run)
        reports/csv/level3_gate_<source>.csv       per-page results
        reports/csv/level3_buckets_<source>.csv    per-page per-bucket chars/tokens
        reports/csv/level3_findings_<source>.csv   edits the in-page oracle rejected
                                                   (need investigation; not proof of harm)

Step 10 writes NO training manifest. Re-run 07 (adds L3 to each page's
ladder) and then 08 (re-validates against the original, freezes targets).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402
from tqdm import tqdm  # noqa: E402

from src.compare.gate import (TokenCounter, evaluate_pair, load_config,  # noqa: E402
                              render_page)
from src.compress.level3_css import level3_page  # noqa: E402
from src.render.harness import RenderHarness  # noqa: E402


def base_for(pid: str, r, tables, prefer: str):
    """(base_path, base_label): step 08's frozen target if it exists (what
    the training set actually contains), else step 07's selection, else the
    original."""
    from src.compress.level3_css import resolve_base
    return resolve_base(ROOT, r.get("source", "webcode2m"), pid, r, prefer, *tables)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="webcode2m")
    ap.add_argument("--k", type=int, default=100)
    ap.add_argument("--on-original", action="store_true")
    ap.add_argument("--base", choices=["final", "selected", "original"], default="final")
    ap.add_argument("--ablate-declarations", action="store_true",
                    help="also delete declarations overridden for every matched element")
    ap.add_argument("--merge-identical", action="store_true",
                    help="merge non-adjacent identical rules (cascade-checked)")
    ap.add_argument("--no-trim-selectors", dest="trim", action="store_false", default=True)
    ap.add_argument("--keep-remote-links", dest="links", action="store_false", default=True)
    ap.add_argument("--no-strict", dest="strict", action="store_false", default=True,
                    help="tolerance-gate acceptance instead of pixel identity (NOT the "
                         "default experiment; the gate cannot see style-detail damage)")
    ap.add_argument("--oracle-budget", type=int, default=200)
    ap.add_argument("--max-ablation", type=int, default=600)
    args = ap.parse_args()
    opts = {"trim_selectors": args.trim, "dead_props": True,
            "drop_remote_links": args.links,
            "ablate_declarations": args.ablate_declarations,
            "merge_identical": args.merge_identical,
            "oracle_budget": args.oracle_budget, "max_ablation": args.max_ablation}

    src = args.source
    manifest = ROOT / "data" / "splits" / f"pilot_{src}_manifest.csv"
    df = pd.read_csv(manifest).head(args.k)
    from src.compress.level3_css import load_base_tables
    tables = load_base_tables(ROOT, src)
    prefer = "original" if args.on_original else args.base

    out_dir = ROOT / "outputs" / "level3" / src
    out_dir.mkdir(parents=True, exist_ok=True)
    work = ROOT / "outputs" / "renders_gate" / src / "l3"
    work.mkdir(parents=True, exist_ok=True)
    rep = ROOT / "reports" / "csv"
    rep.mkdir(parents=True, exist_ok=True)

    cfg = load_config(ROOT / "config" / "gate_config.yaml")
    tk = TokenCounter.get()
    rows, bucket_rows, finding_rows = [], [], []
    t0 = time.time()

    with RenderHarness() as h:
        for _, r in tqdm(df.iterrows(), total=len(df), desc=f"level3[{src}]"):
            pid = str(r["page_id"])
            out_path = out_dir / f"{pid}.html"
            # No stale results: whatever this run decides, yesterday's accepted
            # file must not survive an early exit today.
            out_path.unlink(missing_ok=True)
            row = {"page_id": pid, "source": src}
            try:
                # Original tokens from the FILE, independent of rendering, so a
                # render failure still counts in the corpus denominator.
                orig_html = Path(r["html_path"]).read_text(encoding="utf-8", errors="ignore")
                row["tokens_original"] = tk.count(orig_html)
                base_path, base_label = base_for(pid, r, tables, prefer)
                base_html = base_path.read_text(encoding="utf-8", errors="ignore")
                row.update({"base": base_label, "base_path": str(base_path),
                            "tokens_base": tk.count(base_html)})
            except Exception as e:  # noqa: BLE001
                rows.append({**row, "status": f"base_error: {e}", "accepted": False}); continue
            try:
                art_ref = render_page(h, Path(r["html_path"]), work / f"{pid}__orig.png", tk)
                art_ref.html_text = base_html                 # G1 measures vs the base
                art_ref.tokens = row["tokens_base"]           # image/layout stay the original's
                h.load(base_path)                             # page for the oracle
                res = level3_page(h, base_html, {**opts, "base_dir": base_path.parent})
            except Exception as e:  # noqa: BLE001
                rows.append({**row, "status": f"page_error: {e}", "accepted": False}); continue

            row.update({k: v for k, v in res.items()
                        if k in ("status", "n_blocks", "n_edits_planned", "n_edits_kept",
                                 "oracle_calls", "css_chars_before", "css_chars_after")})
            for f in res.get("blacklist", []):
                finding_rows.append({"page_id": pid, **f})
            if res["status"] != "ok" or res["html"] == base_html:
                rows.append({**row, "status": res["status"] if res["status"] != "ok"
                             else "no_change", "accepted": False, "reduction_pct": 0.0})
                continue

            # Validate the exact file that step 07/08 will consume: write it
            # to its final path, render THAT, delete it unless accepted.
            accepted = False
            try:
                out_path.write_text(res["html"], encoding="utf-8")
                art_c = render_page(h, out_path, work / f"{pid}__l3.png", tk)
                R = evaluate_pair(art_ref, art_c, cfg, page_id=pid)
                shorter = art_c.tokens < row["tokens_base"]   # explicit, config-independent
                R["accepted_strict"] = bool(R["accepted"]) and bool(R["pixel_identical"]) and shorter
                accepted = (bool(R["accepted"]) and shorter
                            and (bool(R["pixel_identical"]) or not args.strict))
                R["accepted"] = accepted
                R["tokens_l3"] = art_c.tokens
            except Exception as e:  # noqa: BLE001
                rows.append({**row, "status": f"gate_error: {e}", "accepted": False}); continue
            finally:
                if not accepted:
                    out_path.unlink(missing_ok=True)

            b = res.get("buckets_chars", {})
            for bucket, chars in b.items():
                bucket_rows.append({"page_id": pid, "bucket": bucket, "chars": chars})
            rows.append({**row, "status": "ok", "buckets": json.dumps(b),
                         **{k: v for k, v in R.items() if k != "page_id"}})
            pd.DataFrame(rows).to_csv(rep / f"level3_gate_{src}.csv", index=False)

    gate_csv = rep / f"level3_gate_{src}.csv"
    pd.DataFrame(rows).to_csv(gate_csv, index=False)
    pd.DataFrame(bucket_rows).to_csv(rep / f"level3_buckets_{src}.csv", index=False)
    pd.DataFrame(finding_rows).to_csv(rep / f"level3_findings_{src}.csv", index=False)

    # ---- report -----------------------------------------------------------
    d = pd.DataFrame(rows)
    print(f"\n[10] ---- Level 3 ({src}, {len(d)} pages, "
          f"{'pixel-identity' if args.strict else 'tolerance-gate'} acceptance) ----")
    print(d["status"].value_counts().to_string())
    if "base" in d:
        print(f"base: {d['base'].value_counts().to_dict()}")
    ok = d[d["status"] == "ok"] if "status" in d else d.iloc[0:0]
    t_orig_all = pd.to_numeric(d.get("tokens_original"), errors="coerce")
    n_unknown = int(t_orig_all.isna().sum())
    if len(ok):
        acc = ok["accepted"].astype(bool)
        red = pd.to_numeric(ok.loc[acc, "reduction_pct"], errors="coerce")
        print(f"\naccepted: {int(acc.sum())}/{len(d)}   pixel-identical among gated: "
              f"{int(ok['pixel_identical'].astype(bool).sum())}/{len(ok)}")
        if len(red):
            print(f"reduction vs base on accepted pages: mean {red.mean():.2f}%  "
                  f"median {red.median():.2f}%  p90 {red.quantile(.9):.2f}%")
        tb = pd.to_numeric(ok.loc[acc, "tokens_base"]).sum()
        tl = pd.to_numeric(ok.loc[acc, "tokens_comp"]).sum()
        to = t_orig_all.sum()
        # Two different quantities, reported as such:
        print(f"L3 INCREMENTAL saving  : {int(tb - tl):,} tokens = {100 * (tb - tl) / max(to, 1):.2f}% "
              f"of all original tokens ({int(to):,}"
              f"{'; ' + str(n_unknown) + ' originals unreadable, denominator incomplete' if n_unknown else ''})")
        base_t = pd.to_numeric(d.get("tokens_base"), errors="coerce")
        final_t = base_t.copy()
        final_t.loc[ok.index[acc]] = pd.to_numeric(ok.loc[acc, "tokens_comp"])
        final_t = final_t.fillna(t_orig_all)          # failed pages: original
        print(f"TOTAL pipeline saving  : {100 * (to - final_t.sum()) / max(to, 1):.2f}% of original "
              f"tokens if step 07/08 adopt these L3 outputs (provisional: step 08 re-validates)")
        bd = pd.DataFrame(bucket_rows)
        if len(bd):
            acc_ids = set(ok.loc[acc, "page_id"].astype(str))
            bd = bd[bd["page_id"].astype(str).isin(acc_ids)]
            tot = max(bd["chars"].sum(), 1)
            print("\nwhere the CSS savings came from (chars, accepted pages):")
            for bucket, g in bd.groupby("bucket"):
                print(f"  {bucket:18s} {100 * g['chars'].sum() / tot:5.1f}%  "
                      f"({int(g['chars'].sum()):,} chars, {g['page_id'].nunique()} pages)")
        print(f"\noracle-rejected edits needing investigation: {len(finding_rows)} "
              f"(level3_findings_{src}.csv; the resolver's blacklist is not exhaustive under its budget)")
    print(f"[10] tokenizer: {tk.name}   wall time {time.time() - t0:.0f}s")
    print(f"[10] wrote {gate_csv}")


if __name__ == "__main__":
    main()