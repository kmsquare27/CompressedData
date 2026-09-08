"""What exactly changed those pixels: Level 1 operator, or which Level 2 edit?

v1 only tested edits whose element OVERLAPPED the diff box. That misses the
common case: an element misclassified as invisible usually has a zero-size
rect at an unrelated position, so the removal that blanked a region never
appeared in the suspect list. This version does two decisive things instead.

  PART 1 -- Level 1 ablation. Starting from the ORIGINAL, apply one L1a
  operator at a time (comments / scripts / attributes / meta+link / CSS
  minification) and render. An operator that introduces the diff on its own
  is the cause. Script removal is the prime suspect: the harness executes
  inline JavaScript, so a page whose script writes into the DOM renders
  differently once the script is gone -- "render-neutral by construction"
  does not hold for that operator.

  PART 2 -- Level 2 delta debugging over the FULL kept-edit set (not just
  overlapping ones), splitting until a minimal responsible subset remains.
  Runs only when Level 1 is clean but Level 2 is not.

Read-only: renders into outputs/renders_gate/<src>/culprit2/, writes one CSV.

Usage:
    python src/tools/find_culprit2.py --source webcode2m \
        --pages we00016 we00091 we00092
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from bs4 import BeautifulSoup, Comment  # noqa: E402
from bs4.element import Stylesheet  # noqa: E402

from src.compare.gate import TokenCounter, load_config, render_page  # noqa: E402
from src.compress.level1_minify import minify_css_text, minify_style_attr  # noqa: E402
from src.render.harness import RenderHarness  # noqa: E402
from src.stress import annotate  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "step06", ROOT / "src" / "pipeline" / "06_run_level2.py")
step06 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(step06)


# ---------------------------------------------------------------------------
# Level 1 operators, individually
# ---------------------------------------------------------------------------
def op_comments(soup):
    for c in soup.find_all(string=lambda t: isinstance(t, Comment)):
        c.extract()


def op_scripts(soup):
    for t in soup.find_all(["script", "noscript"]):
        t.decompose()


def op_attrs(soup):
    css = " ".join(s.get_text() or "" for s in soup.find_all("style"))
    data_styled = "[data-" in css
    for t in soup.find_all(True):
        for a in list(t.attrs):
            al = a.lower()
            if al.startswith("on") or al.startswith("aria-") or al == "title":
                del t.attrs[a]
            elif al.startswith("data-") and not data_styled:
                del t.attrs[a]


def op_head(soup):
    for t in soup.find_all("link"):
        rel = t.get("rel", [])
        rel = " ".join(rel).lower() if isinstance(rel, list) else str(rel).lower()
        if "stylesheet" not in rel:
            t.decompose()
    for t in soup.find_all("meta"):
        keep = (t.has_attr("charset") or str(t.get("name", "")).lower() == "viewport"
                or str(t.get("http-equiv", "")).lower() == "content-type")
        if not keep:
            t.decompose()


def op_css(soup):
    for t in soup.find_all(True):
        if t.has_attr("style") and isinstance(t["style"], str):
            t["style"] = minify_style_attr(t["style"])
    for s in soup.find_all("style"):
        css = s.string if s.string is not None else s.get_text()
        if css:
            new = minify_css_text(css)
            if new != css:
                s.clear(); s.append(Stylesheet(new))


L1_OPS = [("comments", op_comments), ("scripts", op_scripts), ("attributes", op_attrs),
          ("meta+link", op_head), ("css_minify", op_css)]


def apply_one(html: str, fn) -> str:
    soup = BeautifulSoup(html, "lxml")
    fn(soup)
    return str(soup).strip()


def diff_of(a: np.ndarray, b: np.ndarray):
    if a.shape != b.shape:
        return None, None
    m = np.any(a != b, axis=2)
    if not m.any():
        return 0, None
    ys, xs = np.nonzero(m)
    return int(m.sum()), (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="webcode2m")
    ap.add_argument("--pages", nargs="+", required=True)
    ap.add_argument("--max-calls", type=int, default=60, help="renders per page in part 2")
    args = ap.parse_args()
    src = args.source

    man = pd.read_csv(ROOT / "data" / "splits" / f"pilot_{src}_manifest.csv")
    man["page_id"] = man["page_id"].astype(str)
    edits = pd.read_csv(ROOT / "reports" / "csv" / f"level2_edits_{src}.csv")
    edits["page_id"] = edits["page_id"].astype(str)
    l1_dir, l2_dir = ROOT / "outputs" / "level1" / src, ROOT / "outputs" / "level2" / src
    work = ROOT / "outputs" / "renders_gate" / src / "culprit2"
    work.mkdir(parents=True, exist_ok=True)
    load_config(ROOT / "config" / "gate_config.yaml")
    tk = TokenCounter.get()
    rows = []

    with RenderHarness() as h:
        for pid in [str(p) for p in args.pages]:
            r = man[man.page_id == pid].iloc[0]
            orig = Path(r["html_path"])
            html = orig.read_text(encoding="utf-8", errors="ignore")
            print(f"\n=== {pid} " + "=" * 62)
            art_o = render_page(h, orig, work / f"{pid}_orig.png", tk)

            n_l1 = n_l2 = 0
            for stage, p in (("l1", l1_dir / f"{pid}.html"), ("l2", l2_dir / f"{pid}.html")):
                if p.exists():
                    art = render_page(h, p, work / f"{pid}_{stage}.png", tk)
                    n, box = diff_of(art_o.img, art.img)
                    print(f"{stage}: diff px {n} bbox {box}")
                    if stage == "l1":
                        n_l1 = n or 0
                    else:
                        n_l2 = n or 0

            # ---- PART 1: which Level 1 operator? --------------------------
            if n_l1:
                soup_probe = BeautifulSoup(html, "lxml")
                print(f"\nLevel 1 operators applied one at a time to the ORIGINAL "
                      f"(page has {len(soup_probe.find_all(['script', 'noscript']))} script/noscript, "
                      f"{len(soup_probe.find_all(string=lambda t: isinstance(t, Comment)))} comments):")
                for name, fn in L1_OPS:
                    p = work / f"{pid}__op_{name.replace('+', '_')}.html"
                    try:
                        p.write_text(apply_one(html, fn), encoding="utf-8")
                        art = render_page(h, p, work / f"{pid}__op_{name}.png", tk)
                        n, box = diff_of(art_o.img, art.img)
                    except Exception as e:  # noqa: BLE001
                        print(f"  {name:12s} error: {e}"); continue
                    finally:
                        p.unlink(missing_ok=True)
                    tag = "CAUSES THE DIFF" if n else "neutral"
                    print(f"  {name:12s} diff px {n:6d}  bbox {str(box):26s} {tag}")
                    rows.append({"page_id": pid, "part": "l1_op", "name": name,
                                 "diff_px": n, "bbox": str(box)})
                print("  (script removal is not render-neutral when the harness executes "
                      "inline JS that writes into the DOM)")

            # ---- PART 2: which Level 2 edits? -----------------------------
            if n_l2 and n_l2 != n_l1:
                base = l1_dir / f"{pid}.html"
                base = base if base.exists() else orig
                stamped, _ = annotate.stamp_ids(base.read_text(encoding="utf-8", errors="ignore"))
                if stamped is None:
                    print("no <body>; cannot bisect"); continue
                kept = [(row["kind"], int(row["el_id"]))
                        for _, row in edits[(edits.page_id == pid)
                                            & (edits.kept.astype(bool))].iterrows()]
                art_base = render_page(h, base, work / f"{pid}_base.png", tk)
                calls = {"n": 0}

                def diff_with(subset) -> int:
                    calls["n"] += 1
                    p = work / f"{pid}__try.html"
                    p.write_text(step06.apply_edits(stamped, list(subset)), encoding="utf-8")
                    art = render_page(h, p, work / f"{pid}__try.png", tk)
                    n, _ = diff_of(art_base.img, art.img)
                    p.unlink(missing_ok=True)
                    return n or 0

                def isolate(subset):
                    """Smallest subset still producing a difference vs the base."""
                    if len(subset) <= 1 or calls["n"] >= args.max_calls:
                        return subset
                    mid = len(subset) // 2
                    for half in (subset[:mid], subset[mid:]):
                        if half and diff_with(half):
                            return isolate(half)
                    return subset          # interaction between the halves

                print(f"\nbisecting {len(kept)} kept Level 2 edits against the L1 base "
                      f"(diff of the full set: {diff_with(kept)} px):")
                minimal = isolate(kept)
                print(f"  minimal responsible subset: {minimal}   ({calls['n']} renders)")
                stamped_soup = BeautifulSoup(stamped, "lxml")
                for kind, eid in minimal:
                    el = stamped_soup.find(attrs={annotate.STAMP_ATTR: str(eid)})
                    desc = (str(el)[:220] + " ...") if el else "not found"
                    print(f"  {kind} id={eid}: {desc}")
                    rows.append({"page_id": pid, "part": "l2_edit", "name": f"{kind}:{eid}",
                                 "diff_px": n_l2, "bbox": desc[:120]})

    if rows:
        out = ROOT / "reports" / "csv" / f"culprit2_{src}.csv"
        pd.DataFrame(rows).to_csv(out, index=False)
        print(f"\n[culprit2] wrote {out}")


if __name__ == "__main__":
    main()