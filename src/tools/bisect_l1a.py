"""Which L1a sub-transform first changes the render? Measured, not guessed.

Static feature comparison found no culprit (no aria attributes exist on these
pages; !important runs the wrong way). So apply L1a's steps CUMULATIVELY, in
the order level1a() applies them, rendering after each, and report the FIRST
step at which the pixels stop matching the original. That names the offending
transform per page instead of proposing another hypothesis.

Cost: ~11 renders per page. On the 15 rejected pages that is a few minutes.

Usage:
    python tools/bisect_l1a.py --source webcode2m              # rejected pages
    python tools/bisect_l1a.py --source webcode2m --all
    python tools/bisect_l1a.py --source webcode2m --pages we00025 we00047

Output: reports/csv/l1a_bisect_<source>.csv
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
from bs4 import BeautifulSoup, Comment  # noqa: E402
from tqdm import tqdm  # noqa: E402

from src.compress.level1_minify import minify_css_text, minify_style_attr  # noqa: E402
from src.render.harness import RenderHarness  # noqa: E402

try:
    from bs4.element import Stylesheet
except Exception:                                    # older bs4
    Stylesheet = None


# Each step mutates the soup in place. Order mirrors level1a().
def s_comments(soup, ctx):
    for c in soup.find_all(string=lambda t: isinstance(t, Comment)):
        c.extract()


def s_scripts(soup, ctx):
    for t in soup.find_all(["script", "noscript"]):
        t.decompose()


def s_links(soup, ctx):
    for t in soup.find_all("link"):
        rel = t.get("rel", [])
        rel = " ".join(rel).lower() if isinstance(rel, list) else str(rel).lower()
        if "stylesheet" not in rel:
            t.decompose()


def s_meta(soup, ctx):
    for t in soup.find_all("meta"):
        keep = (t.has_attr("charset")
                or str(t.get("name", "")).lower() == "viewport"
                or str(t.get("http-equiv", "")).lower() == "content-type")
        if not keep:
            t.decompose()


def _drop_attrs(soup, pred):
    for t in soup.find_all(True):
        for attr in list(t.attrs):
            if pred(attr.lower()):
                del t.attrs[attr]


def s_on(soup, ctx):
    _drop_attrs(soup, lambda a: a.startswith("on"))


def s_aria(soup, ctx):
    _drop_attrs(soup, lambda a: a.startswith("aria-"))


def s_title(soup, ctx):
    _drop_attrs(soup, lambda a: a == "title")


def s_data(soup, ctx):
    if ctx["data_attr_styled"]:
        return                                        # guarded, same as L1a
    _drop_attrs(soup, lambda a: a.startswith("data-"))


def s_style_attr(soup, ctx):
    for t in soup.find_all(True):
        if t.has_attr("style") and isinstance(t["style"], str):
            new = minify_style_attr(t["style"])
            if new != t["style"]:
                t["style"] = new


def s_style_block(soup, ctx):
    for s in soup.find_all("style"):
        css = s.string if s.string is not None else s.get_text()
        if not css:
            continue
        new = minify_css_text(css)
        if new != css:
            s.clear()
            s.append(Stylesheet(new) if Stylesheet else new)


def s_roundtrip(soup, ctx):
    """No-op CONTROL. Step 1 of any BeautifulSoup-based transform is really
    'parse with lxml and re-serialize', which on malformed real-world HTML
    repairs the DOM: missing html/head/body inserted, unclosed tags closed,
    misplaced content relocated. That alone can change the render. Without
    this control the damage is misattributed to whatever transform happens to
    run first."""
    return


STEPS = [("roundtrip", s_roundtrip),
         ("comments", s_comments), ("scripts", s_scripts), ("links", s_links),
         ("meta", s_meta), ("attr_on", s_on), ("attr_aria", s_aria),
         ("attr_title", s_title), ("attr_data", s_data),
         ("style_attr_min", s_style_attr), ("style_block_min", s_style_block)]


def build(html: str, upto: int) -> str:
    """Apply the first `upto` steps of L1a and serialise."""
    soup = BeautifulSoup(html, "lxml")
    ctx = {"data_attr_styled": "[data-" in " ".join(
        s.get_text() for s in soup.find_all("style"))}
    for _, fn in STEPS[:upto]:
        fn(soup, ctx)
    return str(soup).strip()


def px_diff(a: Path, b: Path) -> int:
    x = np.asarray(Image.open(a).convert("RGB"))
    y = np.asarray(Image.open(b).convert("RGB"))
    if x.shape != y.shape:
        return -1                                     # shape change = big
    return int(np.any(x != y, axis=-1).sum())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="webcode2m")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--pages", nargs="*", default=None)
    args = ap.parse_args()

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
        print(f"[bisect] {len(want)} L1a-rejected pages")
    df = man[man.page_id.isin(want)]

    work = ROOT / "outputs" / "l1a_bisect" / args.source
    work.mkdir(parents=True, exist_ok=True)
    rows = []
    with RenderHarness() as h:
        for _, r in tqdm(df.iterrows(), total=len(df), desc="bisect L1a"):
            pid = str(r["page_id"])
            html = Path(r["html_path"]).read_text(encoding="utf-8", errors="ignore")
            base_png = work / f"{pid}__base.png"
            try:
                h.render(r["html_path"], base_png)
            except Exception as e:                    # noqa: BLE001
                rows.append({"page_id": pid, "first_breaking_step": f"render_error: {e}"})
                continue
            first, per = None, {}
            for i, (name, _) in enumerate(STEPS, start=1):
                p = work / f"{pid}__{i:02d}_{name}.html"
                p.write_text(build(html, i), encoding="utf-8")
                png = work / f"{pid}__{i:02d}_{name}.png"
                try:
                    h.render(p, png)
                    d = px_diff(base_png, png)
                except Exception as e:                # noqa: BLE001
                    d, first = -2, first or f"{name}(error:{e})"
                per[name] = d
                if d != 0 and first is None:
                    first = name
            rows.append({"page_id": pid, "first_breaking_step": first or "none",
                         **{f"px_{k}": v for k, v in per.items()}})

    out = pd.DataFrame(rows)
    rep = ROOT / "reports" / "csv"
    rep.mkdir(parents=True, exist_ok=True)
    out.to_csv(rep / f"l1a_bisect_{args.source}.csv", index=False)

    print(f"\n[bisect] first step at which pixels differ from the original:")
    print(out["first_breaking_step"].value_counts().to_string())
    print("\nper-page (pixels differing; -1 = page size changed):")
    cols = ["page_id", "first_breaking_step"] + [f"px_{n}" for n, _ in STEPS]
    print(out[[c for c in cols if c in out]].to_string(index=False))
    print(f"\n[bisect] wrote {rep / f'l1a_bisect_{args.source}.csv'}")
    print("The step named for most pages is the transform to fix or gate.")


if __name__ == "__main__":
    main()