"""Step 09 -- CSS census: how much of the CSS is dead under the training render?

Same spirit as step 05 (decide before building), same classifier Level 3
uses (compress/level3_css.py), one page load per page, no screenshots. For
every page the browser answers the classification questions and the planner
produces the edit list Level 3 WOULD apply; this script only measures it.

Everything here is a PROPOSED removal candidate: what the planner would
emit, before the in-page oracle, before the render gate. Chars are exact
(source spans minus replacement text). Tokens are exact deltas too:
T(page) - T(page with that bucket's edits applied), and the total is
T(page) - T(page with all edits), so per-bucket tokens need not sum to the
total (tokenization interacts). <link rel=stylesheet> markup is HTML, not
CSS, and is reported separately from the CSS percentages. Two denominators
are printed: measured pages, and ALL input pages (pages without <style>
blocks count with their real page tokens; failed pages count at 0 and are
listed).

The base is step 08's frozen target when it exists (what the training set
actually contains), else step 07's selection, else the original.

Two optional estimates:
  --ablate     in-page declaration ablation on live rules (no renders, but
               ~600 oracle trials per page): the second-order prize
  --merge      the non-adjacent identical-declaration merge; each group is
               oracle-checked ("oracle-passed", not proven safe)
Both are skipped for a page whose planned edit list fails the oracle.

Usage:
    python src/pipeline/09_css_census.py --source webcode2m --n 100
    python src/pipeline/09_css_census.py --source webcode2m --n 30 --ablate --merge

Output: reports/csv/css_census_<source>.csv         one row per page
        reports/csv/css_census_buckets_<source>.csv one row per page x bucket
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402
from tqdm import tqdm  # noqa: E402

from src.compare.gate import TokenCounter  # noqa: E402
from src.compress import level3_css as L3  # noqa: E402
from src.render.harness import RenderHarness  # noqa: E402

CSS_BUCKETS = ("unmatched", "stateful", "selector_trim", "media_nomatch",
                "supports_nomatch", "block_emptied", "keyframes",
                "font_face_remote", "import", "page", "dead_props")


def count_rules(rules) -> tuple[int, int]:
    n, s = 0, 0
    for r in rules:
        n += 1
        s += r.kind == "style"
        a, b = count_rules(r.children)
        n += a; s += b
    return n, s


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="webcode2m")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--base", choices=["final", "selected", "original"], default="final",
                    help="final = step 08 targets (default), selected = step 07, original")
    ap.add_argument("--on-original", action="store_true", help="alias for --base original")
    ap.add_argument("--ablate", action="store_true")
    ap.add_argument("--merge", action="store_true")
    args = ap.parse_args()
    src = args.source
    prefer = "original" if args.on_original else args.base

    manifest = ROOT / "data" / "splits" / f"pilot_{src}_manifest.csv"
    df = pd.read_csv(manifest).head(args.n)
    final_df, sel_df = L3.load_base_tables(ROOT, src)
    tk = TokenCounter.get()
    opts = {"trim_selectors": True, "dead_props": True}
    rows, brows = [], []

    with RenderHarness() as h:
        for _, r in tqdm(df.iterrows(), total=len(df), desc=f"css census[{src}]"):
            pid = str(r["page_id"])
            row = {"page_id": pid}
            try:
                path, label = L3.resolve_base(ROOT, src, pid, r, prefer, final_df, sel_df)
                html = path.read_text(encoding="utf-8", errors="ignore")
                row.update({"base": label, "base_path": str(path),
                            "tokens_page": tk.count(html), "chars_page": len(html)})
                blocks = L3.find_style_blocks(html)
                css_all = [b.css for b in blocks]
                row.update({"n_style_blocks": len(blocks),
                            "chars_css": sum(len(c) for c in css_all),
                            "tokens_css": sum(tk.count(c) for c in css_all)})
                link_spans = L3.remote_link_edits(html)
                per_bucket_edits: dict[str, list] = {}       # bucket -> [(block_idx, Edit)]
                n_rules = n_style = 0
                ans = {"selectors": {}, "media": {}, "supports": {}}
                q = {"selectors": {}, "media": {}, "supports": {}}
                all_rules = []
                if blocks:
                    h.load(path)
                    dom = h.page.evaluate(L3.GET_STYLES_JS)
                    if len(dom) != len(blocks) or any(d != c for d, c in zip(dom, css_all)):
                        rows.append({**row, "status": "style_mismatch"}); continue
                    all_rules = [L3.parse_rules(c) for c in css_all]
                    for rules in all_rules:
                        L3.collect_questions(rules, q)
                    ans = L3.ask_browser(h, q)
                    page_opts = {**opts, "base_dir": path.parent}
                    for bi, (rules, css) in enumerate(zip(all_rules, css_all)):
                        a, b = count_rules(rules); n_rules += a; n_style += b
                        for e in L3.plan_rules(rules, css, ans, page_opts)[0]:
                            per_bucket_edits.setdefault(e.bucket, []).append((bi, e))
            except Exception as e:  # noqa: BLE001
                rows.append({**row, "status": f"page_error: {e}"}); continue

            def proposed_html(bucket_filter=None) -> str:
                per = [[] for _ in blocks]
                for bucket, lst in per_bucket_edits.items():
                    if bucket_filter is None or bucket == bucket_filter:
                        for bi, e in lst:
                            per[bi].append(e)
                out = L3.splice_blocks(html, blocks, [L3.apply_edits(c, eds) for c, eds in zip(css_all, per)])
                if bucket_filter in (None, "link_remote"):
                    spans = L3.remote_link_edits(out)
                    for a, b in sorted(spans, reverse=True):
                        out = out[:a] + out[b:]
                return out

            t_page = row["tokens_page"]
            css_chars = css_tokens = 0
            for bucket, lst in per_bucket_edits.items():
                chars = sum(e.chars for _, e in lst)
                toks = t_page - tk.count(proposed_html(bucket))
                brows.append({"page_id": pid, "bucket": bucket, "chars": chars, "tokens": toks,
                              "n_edits": len(lst)})
                row[f"chars_{bucket}"] = chars; row[f"tokens_{bucket}"] = toks
                if bucket in CSS_BUCKETS:
                    css_chars += chars; css_tokens += toks
            if link_spans:
                lc = sum(b - a for a, b in link_spans)
                lt = t_page - tk.count(proposed_html("link_remote"))
                brows.append({"page_id": pid, "bucket": "link_remote", "chars": lc, "tokens": lt,
                              "n_edits": len(link_spans)})
                row.update({"chars_link_remote": lc, "tokens_link_remote": lt})
            total_tokens = t_page - tk.count(proposed_html(None))
            row.update({"status": "ok" if blocks else "no_style_blocks",
                        "n_rules": n_rules, "n_style_rules": n_style,
                        "n_selectors_asked": len(q["selectors"]),
                        "n_selectors_unmatched": sum(1 for v in ans["selectors"].values() if v == 0),
                        "n_selectors_invalid": sum(1 for v in ans["selectors"].values() if v == -1),
                        "css_candidate_chars": css_chars, "css_candidate_tokens": css_tokens,
                        "css_candidate_pct_of_css_chars": round(100 * css_chars / max(row["chars_css"], 1), 2),
                        "proposed_tokens_saved": total_tokens,
                        "proposed_pct_of_page_tokens": round(100 * total_tokens / max(t_page, 1), 2)})

            if (args.ablate or args.merge) and blocks:
                try:
                    h.oracle_baseline()
                    texts = L3.find_style_blocks(proposed_html(None))
                    texts = [b.css for b in texts]
                    ok, res = L3.oracle_same(h, texts)
                    row["planned_edits_oracle_ok"] = ok
                    if not ok:
                        row["planned_edits_oracle_first"] = res.get("first", "")
                        row["optional_skipped"] = "planned edits failed the oracle"
                    else:
                        if args.ablate:
                            t2, log = L3.ablate_declarations(h, texts, 600)
                            row.update({"ablation_trials": len(log),
                                        "ablation_kept": sum(1 for l in log if l["kept"]),
                                        "chars_decl_ablation": sum(l["chars"] for l in log if l["kept"]),
                                        "tokens_decl_ablation": sum(tk.count(t) for t in texts)
                                        - sum(tk.count(t) for t in t2)})
                            texts = t2
                        if args.merge:
                            t3, log = L3.merge_identical_rules(h, texts)
                            row.update({"merge_groups": len(log),
                                        "merge_oracle_passed": sum(1 for l in log if l["kept"]),
                                        "chars_merge_identical": sum(l["chars"] for l in log if l["kept"]),
                                        "tokens_merge_identical": sum(tk.count(t) for t in texts)
                                        - sum(tk.count(t) for t in t3)})
                    h.page.evaluate(L3.SET_STYLES_JS, css_all)
                except Exception as e:  # noqa: BLE001
                    row["oracle_error"] = str(e)
            rows.append(row)

    out = pd.DataFrame(rows)
    rep = ROOT / "reports" / "csv"; rep.mkdir(parents=True, exist_ok=True)
    out.to_csv(rep / f"css_census_{src}.csv", index=False)
    bd = pd.DataFrame(brows)
    bd.to_csv(rep / f"css_census_buckets_{src}.csv", index=False)

    measured = out[out["status"].isin(["ok", "no_style_blocks"])]
    ok = out[out["status"] == "ok"]
    failed = out[~out["status"].isin(["ok", "no_style_blocks"])]
    print(f"\n[09] ---- CSS census: {src} -- PROPOSED removal candidates under the "
          f"training render (pre-oracle, pre-gate) ----")
    print(f"pages: {len(out)} input, {len(ok)} with <style> blocks measured, "
          f"{int((out['status'] == 'no_style_blocks').sum())} without <style> blocks, "
          f"{len(failed)} failed (counted at 0 in the all-inputs figure)")
    if "base" in out:
        print(f"base: {out['base'].value_counts().to_dict()}")
    if len(failed):
        print(failed["status"].value_counts().to_string())
    if not len(ok):
        return
    tot_css = ok["tokens_css"].sum()
    tot_ok = ok["tokens_page"].sum()                    # pages with <style> blocks
    tot_all = measured["tokens_page"].sum()             # + pages without <style> blocks
    tot_inputs = tot_all + failed.get("tokens_page", pd.Series(dtype=float)).fillna(0).sum()
    print(f"\nCSS share of tokens (measured pages): {100 * tot_css / max(tot_ok, 1):.1f}%   "
          f"rules/page median {ok['n_rules'].median():.0f}, selectors asked "
          f"{int(ok['n_selectors_asked'].sum())}, unmatched {int(ok['n_selectors_unmatched'].sum())}, "
          f"invalid(kept) {int(ok['n_selectors_invalid'].sum())}")
    note = {"unmatched": "selector matches nothing",
            "stateful": ":hover/:focus/... and matches nothing in the captured state",
            "selector_trim": "dead items of selector lists",
            "media_nomatch": "@media false at 1280px/light/static",
            "supports_nomatch": "@supports false", "block_emptied": "conditional block fully dead",
            "keyframes": "animations frozen", "font_face_remote": "no reachable source",
            "import": "unreachable target", "page": "print only",
            "dead_props": "cursor/transition/animation/...",
            "link_remote": "<link rel=stylesheet http(s)> (HTML, not CSS)"}
    print(f"\n{'bucket':<18}{'tokens':>9}{'% of CSS':>10}{'% of pages':>11}{'pages':>7}   note")
    if len(bd):
        for bucket, g in bd.groupby("bucket"):
            t = g["tokens"].sum()
            css_pct = f"{100 * t / max(tot_css, 1):>9.1f}%" if bucket in CSS_BUCKETS else f"{'-':>10}"
            print(f"{bucket:<18}{int(t):>9}{css_pct}{100 * t / max(tot_all, 1):>10.1f}%"
                  f"{g['page_id'].nunique():>7}   {note.get(bucket, '')}")
    css_t = ok["css_candidate_tokens"].sum()
    prop = measured["proposed_tokens_saved"].fillna(0).sum()
    print(f"\nCSS removal candidates : {int(css_t):,} tokens = {100 * css_t / max(tot_css, 1):.1f}% of CSS tokens "
          f"= {100 * css_t / max(tot_all, 1):.1f}% of measured pages' tokens")
    print(f"PROPOSED saving, exact : {int(prop):,} tokens = {100 * prop / max(tot_all, 1):.1f}% of measured pages"
          f" = {100 * prop / max(tot_inputs, 1):.1f}% of ALL input pages (failed at 0)"
          f"   (per <style> page: median {ok['proposed_pct_of_page_tokens'].median():.1f}%, "
          f"p90 {ok['proposed_pct_of_page_tokens'].quantile(.9):.1f}%)")
    if "planned_edits_oracle_ok" in ok:
        po = ok["planned_edits_oracle_ok"].dropna()
        print(f"planned edit lists passing the in-page oracle whole: {int(po.astype(bool).sum())}/{len(po)}"
              f"  (failures are classification bugs: see planned_edits_oracle_first)")
    if "tokens_decl_ablation" in ok:
        a = ok["tokens_decl_ablation"].fillna(0).sum()
        print(f"second-order, oracle-passed (--ablate): {int(a):,} tokens = "
              f"{100 * a / max(tot_all, 1):.1f}% of measured pages; trials "
              f"{int(ok['ablation_trials'].fillna(0).sum())}, kept {int(ok['ablation_kept'].fillna(0).sum())}")
    if "tokens_merge_identical" in ok:
        m = ok["tokens_merge_identical"].fillna(0).sum()
        print(f"identical-rule merge, oracle-passed (--merge): {int(m):,} tokens = "
              f"{100 * m / max(tot_all, 1):.2f}% of measured pages; groups "
              f"{int(ok['merge_groups'].fillna(0).sum())}, passed {int(ok['merge_oracle_passed'].fillna(0).sum())}")
    print("\nThese are candidates. Level 3 (step 10) applies them, checks the oracle, renders, "
          "and requires pixel identity + G1 against the original.")
    print(f"\n[09] wrote {rep / f'css_census_{src}.csv'}")


if __name__ == "__main__":
    main()