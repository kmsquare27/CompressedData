"""Which part of L1a is not render-neutral?

L1a strips comments, script/noscript, non-stylesheet <link>, most <meta>,
and the attributes on*/aria-*/title/data-*, then minifies CSS. Scripts are
ruled out (WebCode2M has none). This scores the remaining suspects
statically on the pages the gate REJECTED vs the ones it accepted, so the
next step is a targeted fix rather than another guess.

The strongest single hypothesis it tests: L1a guards `data-*` removal behind
a "[data-" check on the page's CSS, but strips `aria-*` unconditionally.
A rule like `[aria-hidden="true"]{display:none}` is common, and stripping
the attribute would make hidden content VISIBLE -- which matches the two
catastrophic pages gaining 29 boxes and 1278px of height.

Usage:  python tools/diagnose_l1a.py --source webcode2m
"""
from __future__ import annotations

import argparse
import pathlib
import re
import sys

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]


def features(html: str) -> dict:
    soup_css = " ".join(re.findall(r"<style[^>]*>(.*?)</style>", html, re.S | re.I))
    head = html[:html.lower().find("</head>") + 7] if "</head>" in html.lower() else html[:4000]
    return {
        # attribute selectors naming an attribute L1a deletes
        "css_aria_sel": len(re.findall(r"\[\s*aria-", soup_css, re.I)),
        "css_title_sel": len(re.findall(r"\[\s*title", soup_css, re.I)),
        "css_data_sel": len(re.findall(r"\[\s*data-", soup_css, re.I)),
        "css_on_sel": len(re.findall(r"\[\s*on[a-z]+", soup_css, re.I)),
        # attributes actually present in the document
        "has_aria_attr": len(re.findall(r"\saria-[a-z-]+\s*=", html, re.I)),
        "has_title_attr": len(re.findall(r"\stitle\s*=", html, re.I)),
        # head elements L1a prunes
        "meta_color_scheme": len(re.findall(r"<meta[^>]+color-scheme", head, re.I)),
        "n_meta": len(re.findall(r"<meta", head, re.I)),
        "n_link_nonstyle": len([m for m in re.findall(r"<link[^>]*>", head, re.I)
                                if "stylesheet" not in m.lower()]),
        # CSS features lightningcss rewrites
        "css_chars": len(soup_css),
        "css_important": len(re.findall(r"!important", soup_css, re.I)),
        "css_media": len(re.findall(r"@media", soup_css, re.I)),
        "css_var": len(re.findall(r"var\(", soup_css, re.I)),
        "css_escape": len(re.findall(r"\\[0-9a-fA-F]{2,6}", soup_css)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="webcode2m")
    args = ap.parse_args()

    s = pd.read_csv(ROOT / "reports" / "csv" / f"level1_stages_{args.source}.csv")
    man = pd.read_csv(ROOT / "data" / "splits" / f"pilot_{args.source}_manifest.csv")
    man["page_id"] = man["page_id"].astype(str)
    a = s[(s.stage == "l1a") & (s.status == "ok")].copy()
    a["page_id"] = a["page_id"].astype(str)
    a["ok"] = a["accepted"].astype(str).str.lower().isin(["true", "1"])
    a = a.merge(man[["page_id", "html_path"]], on="page_id", how="left")

    rows = []
    for _, r in a.iterrows():
        try:
            html = pathlib.Path(r["html_path"]).read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        rows.append({"page_id": r["page_id"], "ok": bool(r["ok"]), **features(html)})
    f = pd.DataFrame(rows)
    if f.empty:
        print("[diag] no pages read"); return

    print(f"\n=== L1a: {int((~f.ok).sum())} rejected vs {int(f.ok.sum())} accepted ===")
    print(f"{'feature':<20}{'rejected':>12}{'accepted':>12}{'signal':>10}")
    cols = [c for c in f.columns if c not in ("page_id", "ok")]
    for c in cols:
        rj, ac = f.loc[~f.ok, c], f.loc[f.ok, c]
        # share of pages where the feature is present at all
        pr, pa = (rj > 0).mean(), (ac > 0).mean()
        flag = ""
        if pr >= 0.5 and pr > pa * 2:
            flag = "  <== SUSPECT"
        elif pr > pa + 0.25:
            flag = "  <- elevated"
        print(f"{c:<20}{100*pr:>11.0f}%{100*pa:>11.0f}%{flag:>10}")

    print("\n--- per-page detail for the rejected set ---")
    print(f.loc[~f.ok, ["page_id", "css_aria_sel", "has_aria_attr", "css_data_sel",
                        "meta_color_scheme", "n_link_nonstyle", "css_escape",
                        "css_var", "css_media"]].to_string(index=False))
    print("\nRead: a feature present on most REJECTED pages and few ACCEPTED ones "
          "is the likely culprit. If css_aria_sel is the suspect, the fix is one "
          "line: guard aria-* removal the same way data-* is already guarded.")


if __name__ == "__main__":
    main()