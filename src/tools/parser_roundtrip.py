"""Which BeautifulSoup parser round-trips WITHOUT changing the render?

bisect_l1a.py showed that on 14 of 15 L1a-rejected pages the damage comes
from the no-op control: parse with lxml, serialize, done. None of L1a's ten
transforms adds a single pixel beyond that. So the question is not which
transform to fix but which parser to parse with.

Statically, html.parser is far more faithful than lxml (it does not insert
html/head/body, relocate content out of <head>, or restructure form-in-table).
But it is not obviously better for our purpose: on `<div><p>one<div>two</div>`
it emits `<p>one<div>two</div></p>`, nesting the div INSIDE the p, where a
browser implicitly closes the p first. Faithful to the source text is not the
same as faithful to the browser's DOM.

What matters is only this: does the round-tripped file RENDER the same as the
original? So render it. Original once, then one round-trip per parser.

Usage:
    python tools/parser_roundtrip.py --source webcode2m           # rejected pages
    python tools/parser_roundtrip.py --source webcode2m --all
    python tools/parser_roundtrip.py --source webcode2m --pages we00025 we00078

Output: reports/csv/parser_roundtrip_<source>.csv
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402
from bs4 import BeautifulSoup  # noqa: E402
from tqdm import tqdm  # noqa: E402

from src.render.harness import RenderHarness  # noqa: E402

warnings.filterwarnings("ignore")

PARSERS = ["lxml", "html.parser", "html5lib"]


def available(p: str) -> bool:
    try:
        BeautifulSoup("<p>x</p>", p)
        return True
    except Exception:
        return False


def px_diff(a: Path, b: Path):
    x = np.asarray(Image.open(a).convert("RGB"))
    y = np.asarray(Image.open(b).convert("RGB"))
    if x.shape != y.shape:
        return -1, f"{x.shape[1]}x{x.shape[0]} vs {y.shape[1]}x{y.shape[0]}"
    return int(np.any(x != y, axis=-1).sum()), ""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="webcode2m")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--pages", nargs="*", default=None)
    args = ap.parse_args()

    parsers = [p for p in PARSERS if available(p)]
    missing = [p for p in PARSERS if p not in parsers]
    if missing:
        print(f"[parser] not installed, skipping: {missing} "
              f"(pip install html5lib)")

    man = pd.read_csv(ROOT / "data" / "splits" / f"pilot_{args.source}_manifest.csv")
    man["page_id"] = man["page_id"].astype(str)
    if args.pages:
        want = set(args.pages)
    elif args.all:
        want = set(man["page_id"])
    else:
        st = pd.read_csv(ROOT / "reports" / "csv" / f"level1_stages_{args.source}.csv")
        a = st[(st.stage == "l1a") & (st.status == "ok")].copy()
        a["ok"] = a["accepted"].astype(str).str.lower().isin(["true", "1"])
        want = set(a.loc[~a.ok, "page_id"].astype(str))
        print(f"[parser] {len(want)} L1a-rejected pages")
    df = man[man.page_id.isin(want)]

    work = ROOT / "outputs" / "parser_roundtrip" / args.source
    work.mkdir(parents=True, exist_ok=True)
    rows = []
    with RenderHarness() as h:
        for _, r in tqdm(df.iterrows(), total=len(df), desc="parser round-trip"):
            pid = str(r["page_id"])
            src = Path(r["html_path"])
            html = src.read_text(encoding="utf-8", errors="ignore")
            base = work / f"{pid}__orig.png"
            row = {"page_id": pid, "chars": len(html)}
            try:
                h.render(src, base)
            except Exception as e:  # noqa: BLE001
                rows.append({**row, "error": str(e)}); continue
            for p in parsers:
                col = p.replace(".", "_")
                try:
                    out = str(BeautifulSoup(html, p))
                    f = work / f"{pid}__{col}.html"
                    f.write_text(out, encoding="utf-8")
                    png = work / f"{pid}__{col}.png"
                    h.render(f, png)
                    d, note = px_diff(base, png)
                    row[f"px_{col}"] = d
                    row[f"note_{col}"] = note
                    row[f"bytes_{col}"] = len(out)
                except Exception as e:  # noqa: BLE001
                    row[f"px_{col}"] = -2
                    row[f"note_{col}"] = str(e)[:60]
            rows.append(row)

    out = pd.DataFrame(rows)
    rep = ROOT / "reports" / "csv"
    rep.mkdir(parents=True, exist_ok=True)
    out.to_csv(rep / f"parser_roundtrip_{args.source}.csv", index=False)

    print("\n=== pixels differing after a pure round-trip (0 = faithful, "
          "-1 = page size changed) ===")
    cols = ["page_id"] + [f"px_{p.replace('.', '_')}" for p in parsers]
    print(out[[c for c in cols if c in out]].to_string(index=False))
    print("\n=== how many pages each parser round-trips with ZERO pixel change ===")
    for p in parsers:
        c = f"px_{p.replace('.', '_')}"
        if c in out:
            n = int((out[c] == 0).sum())
            print(f"  {p:<12} {n}/{len(out)} faithful")
    print("\nIf a parser is faithful on most of these pages, switching "
          "BeautifulSoup(html, 'lxml') to it in compress/level1_minify.py and "
          "stress/annotate.py (stamp_ids) recovers them at no cost to the "
          "transforms, which the bisect already showed are neutral.")


if __name__ == "__main__":
    main()