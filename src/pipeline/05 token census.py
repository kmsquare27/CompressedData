"""Step 05 -- token census: where do the tokens actually live?

Decides where compression effort is worth spending BEFORE any of it is built.
Levels 1-3 attack different mass:

    L1   comments, scripts, whitespace          (subtractive, built)
    L1.5 unused CSS rules                       (subtractive, not built)
    L2   invisible subtrees, wrapper nesting    (subtractive, not built)
    L3   restyle from computed styles           (rewrites the page)

If CSS is the dominant mass, a structure-only pipeline (L1+L2) is capped no
matter how well it is engineered. If wrapper nesting dominates, L2 carries the
paper and CSS work is a distraction. This script answers which, in an hour,
against the SAME tokenizer the gate uses -- so the numbers are comparable with
every other figure in the project.

Method: every character of the raw HTML is attributed to exactly one category
by non-overlapping span masking, so the character decomposition is exact and
sums to the file. Token counts per category are then measured by tokenizing
each category's extracted text; those are APPROXIMATE and need not sum to the
document total, because tokenizers are context-dependent (a fragment tokenizes
differently in isolation than in place). Both are reported: characters for
exactness, tokens for the number the paper is about.

Usage:
    python src/pipeline/05_token_census.py --source webcode2m --n 20

Output: reports/csv/token_census_<source>.csv  (one row per page)
        plus a printed summary with per-bucket breakdown.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from tqdm import tqdm  # noqa: E402

from src.compare.gate import TokenCounter  # noqa: E402

# Categories, in the order they are reported.
CATS = ["text", "css", "class_attrs", "style_attrs", "other_attrs",
        "tags", "comments", "scripts", "whitespace"]

RE_COMMENT = re.compile(r"<!--.*?-->", re.S)
RE_SCRIPT = re.compile(r"<script\b[^>]*>.*?</script>", re.S | re.I)
RE_STYLE = re.compile(r"(<style\b[^>]*>)(.*?)(</style>)", re.S | re.I)
RE_TAG = re.compile(r"<[^>]*>", re.S)
RE_ATTR = re.compile(r"""([A-Za-z_:][-\w:.]*)\s*=\s*("[^"]*"|'[^']*'|[^\s>]+)""")


def attribute_spans(html: str) -> np.ndarray:
    """Label every character with exactly one category. Later passes never
    overwrite earlier ones, so the decomposition stays disjoint and total."""
    n = len(html)
    lab = np.full(n, -1, dtype=np.int8)
    idx = {c: i for i, c in enumerate(CATS)}

    def mark(a, b, cat):
        seg = lab[a:b]
        seg[seg == -1] = idx[cat]

    for m in RE_COMMENT.finditer(html):
        mark(m.start(), m.end(), "comments")
    for m in RE_SCRIPT.finditer(html):
        mark(m.start(), m.end(), "scripts")
    for m in RE_STYLE.finditer(html):
        mark(m.start(1), m.end(1), "tags")
        mark(m.start(2), m.end(2), "css")        # the stylesheet itself
        mark(m.start(3), m.end(3), "tags")

    # Remaining tags: attribute values split out, the rest is structural.
    for m in RE_TAG.finditer(html):
        if lab[m.start()] != -1:
            continue
        inner = m.group(0)
        base = m.start()
        for a in RE_ATTR.finditer(inner):
            name = a.group(1).lower()
            cat = ("class_attrs" if name == "class"
                   else "style_attrs" if name == "style"
                   else "other_attrs")
            mark(base + a.start(), base + a.end(), cat)
        mark(m.start(), m.end(), "tags")

    # Whatever is left is document text; split pure whitespace out of it.
    for m in re.finditer(r"\s+", html):
        if lab[m.start()] == -1:
            mark(m.start(), m.end(), "whitespace")
    lab[lab == -1] = idx["text"]
    return lab


def census_page(html: str, tk: TokenCounter) -> dict:
    lab = attribute_spans(html)
    idx = {c: i for i, c in enumerate(CATS)}
    arr = np.frombuffer(html.encode("utf-8", "ignore")[:0], dtype=np.uint8)  # noqa: F841
    out = {"chars_total": len(html), "tokens_total": tk.count(html)}
    for c in CATS:
        sel = np.flatnonzero(lab == idx[c])
        out[f"chars_{c}"] = int(sel.size)
        # Contiguous runs keep the fragment realistic for the tokenizer.
        if sel.size:
            cuts = np.flatnonzero(np.diff(sel) > 1)
            starts = np.concatenate(([0], cuts + 1))
            ends = np.concatenate((cuts, [sel.size - 1]))
            frag = "".join(html[sel[s]:sel[e] + 1] for s, e in zip(starts, ends))
        else:
            frag = ""
        out[f"tokens_{c}"] = tk.count(frag) if frag else 0
    return out


def structure_stats(html: str) -> dict:
    """Cheap structural signals that predict L2 yield, without a browser.

    Wrapper candidates are div/span carrying no direct text -- the population
    L2's collapse pass draws from. This is an UPPER BOUND: the browser will
    disqualify many for padding, background, flex parentage and so on. It
    tells you the size of the prize, not the prize."""
    try:
        from bs4 import BeautifulSoup
    except Exception:
        return {}
    soup = BeautifulSoup(html, "lxml")
    body = soup.body or soup
    els = body.find_all(True)
    wrappers, depth_sum, maxdepth = 0, 0, 0
    for el in els:
        if el.name in ("div", "span"):
            direct = any(getattr(c, "name", None) is None and str(c).strip()
                         for c in el.children)
            if not direct and len(el.find_all(True, recursive=False)) >= 1:
                wrappers += 1
        d = len(list(el.parents))
        depth_sum += d
        maxdepth = max(maxdepth, d)
    return {"n_elements": len(els), "n_wrapper_candidates": wrappers,
            "depth_mean": round(depth_sum / max(len(els), 1), 2),
            "depth_max": maxdepth}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="webcode2m")
    ap.add_argument("--n", type=int, default=20)
    args = ap.parse_args()

    manifest = ROOT / "data" / "splits" / f"pilot_{args.source}_manifest.csv"
    if not manifest.exists():
        print(f"[05] no manifest at {manifest}"); return
    df = pd.read_csv(manifest).head(args.n)
    tk = TokenCounter.get()

    rows = []
    for _, r in tqdm(df.iterrows(), total=len(df), desc=f"census[{args.source}]"):
        html = Path(r["html_path"]).read_text(encoding="utf-8", errors="ignore")
        row = {"page_id": r["page_id"], **census_page(html, tk),
               **structure_stats(html)}
        rows.append(row)

    out = pd.DataFrame(rows)
    rep = ROOT / "reports" / "csv"; rep.mkdir(parents=True, exist_ok=True)
    out_csv = rep / f"token_census_{args.source}.csv"
    out.to_csv(out_csv, index=False)

    tot = out["tokens_total"].sum()
    print(f"\n[05] ---- token census: {args.source} ({len(out)} pages) ----")
    print(f"median page: {out['tokens_total'].median():.0f} tokens "
          f"(min {out['tokens_total'].min():.0f}, "
          f"max {out['tokens_total'].max():.0f})\n")
    print(f"{'category':<14}{'% of tokens':>12}{'median/page':>14}   attacked by")
    owner = {"css": "L1.5 unused-rule pruning / L3 restyle",
             "class_attrs": "L3 restyle (dedup into short classes)",
             "style_attrs": "L3 restyle",
             "tags": "L2 wrapper collapse + invisible subtrees",
             "text": "-- never touch (G3 protects it)",
             "comments": "L1 (already removed)",
             "scripts": "L1 (already removed)",
             "whitespace": "L1 (already minified)",
             "other_attrs": "L1 (partial: tracking/meta attrs)"}
    for c in sorted(CATS, key=lambda c: -out[f"tokens_{c}"].sum()):
        share = 100.0 * out[f"tokens_{c}"].sum() / max(tot, 1)
        print(f"{c:<14}{share:>11.1f}%{out[f'tokens_{c}'].median():>14.0f}"
              f"   {owner[c]}")

    if "n_wrapper_candidates" in out:
        print(f"\nstructure: {out['n_elements'].median():.0f} elements/page, "
              f"{out['n_wrapper_candidates'].median():.0f} wrapper candidates "
              f"(upper bound for L2), mean depth {out['depth_mean'].median():.1f}")

    struct = 100.0 * out["tokens_tags"].sum() / max(tot, 1)
    style = 100.0 * (out["tokens_css"].sum() + out["tokens_class_attrs"].sum()
                     + out["tokens_style_attrs"].sum()) / max(tot, 1)
    print(f"\nstructure mass (L2's ceiling): {struct:.1f}%   "
          f"style mass (CSS+class+inline): {style:.1f}%")
    if style > struct * 1.5:
        print("-> style dominates. L2 alone is capped; consider L1.5 unused-CSS "
              "pruning (subtractive, no page rewrite) before concluding.")
    elif struct > style * 1.5:
        print("-> structure dominates. L2 is the right investment; CSS work "
              "would be a distraction.")
    else:
        print("-> comparable. L2 is worth building; measure it, then decide "
              "whether CSS work is needed to reach a reportable total.")
    print(f"\n[05] wrote {out_csv}")


if __name__ == "__main__":
    main()