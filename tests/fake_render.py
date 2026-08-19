"""A deterministic mini layout engine that mimics the RenderHarness API
(render() -> layout dict + PNG; page.evaluate(ANNOTATE_JS) -> meta), so the
FULL stress-test runner loop can be integration-tested without a browser.

Not a browser: a tiny document-order block layout where inline styles
(display:none, margin-left, padding, font-size, background-color) have the
obvious effects and text content determines fill color -- enough that every
mutation operator produces its EXPECTED pixel/box signature (deletion ->
unmatched block; shift -> center_shift == px; recolor -> region dE00 tracks
severity; safe edits -> pixel-identical)."""
from __future__ import annotations

import zlib
from pathlib import Path

import numpy as np
from bs4 import BeautifulSoup, Comment, NavigableString
from PIL import Image

import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.stress.annotate import STAMP_ATTR  # noqa: E402
from src.stress.mutations import parse_css_color  # noqa: E402

W = 640


def _px(v) -> float:
    if not v:
        return 0.0
    try:
        return float(str(v).replace("!important", "").replace("px", "").strip())
    except ValueError:
        return 0.0


def _sty(el) -> dict:
    d = {}
    for part in (el.get("style") or "").split(";"):
        if ":" in part:
            k, v = part.split(":", 1)
            d[k.strip().lower()] = v.replace("!important", "").strip()
    return d


def _hash_color(text: str):
    h = zlib.crc32(text.encode("utf-8"))
    return (60 + h % 160, 60 + (h >> 8) % 160, 60 + (h >> 16) % 160)


def fake_engine(html: str):
    soup = BeautifulSoup(html, "lxml")
    body = soup.body
    rects, boxes, texts = [], [], []
    own_box: dict = {}
    hid_ids: set = set()
    y = [10]

    def walk(el, ml, hidden):
        st = _sty(el)
        if st.get("display") == "none":
            hidden = True
        sid = el.get(STAMP_ATTR)
        sid = int(sid) if sid is not None else None
        if hidden and sid is not None:
            hid_ids.add(sid)
        ml = ml + _px(st.get("margin-left"))
        pad = _px(st.get("padding"))
        if pad and not hidden:
            y[0] += pad
        direct = " ".join("".join(
            str(c) for c in el.children
            if isinstance(c, NavigableString) and not isinstance(c, Comment)
        ).split())
        fs = _px(st.get("font-size")) or 16.0
        bg = parse_css_color(st.get("background-color")
                             or st.get("background") or "")
        drew = None
        if not hidden:
            x = 10 + int(ml + pad)
            if direct:
                h = max(8, int(round(fs * 1.3)))
                w = min(W - x - 10, max(8, int(len(direct) * fs * 0.55)))
                color = (tuple(int(v) for v in bg[:3])
                         if bg and bg[3] > 0.01 else _hash_color(direct))
                drew = (x, y[0], w, h)
                rects.append((*drew, color))
                boxes.append({"text": direct[:120], "tag": el.name.upper(),
                              "x": x, "y": y[0], "w": w, "h": h})
                texts.append(direct)
                y[0] += h + 6
            elif el.name == "img":
                drew = (x, y[0], 60, 40)
                rects.append((*drew, (150, 150, 150)))
                boxes.append({"text": "", "tag": "IMG", "x": x, "y": y[0],
                              "w": 60, "h": 40})
                y[0] += 46
            elif bg and bg[3] > 0.01 and not el.find_all(True):
                drew = (x, y[0], 200, 30)
                rects.append((*drew, tuple(int(v) for v in bg[:3])))
                boxes.append({"text": "", "tag": el.name.upper(), "x": x,
                              "y": y[0], "w": 200, "h": 30})
                y[0] += 36
        if sid is not None:
            own_box[sid] = drew
        for c in el.find_all(True, recursive=False):
            walk(c, ml, hidden)
        if pad and not hidden:
            y[0] += pad

    if body is not None:
        walk(body, 0.0, False)
    docH = int(y[0] + 10)
    img = np.full((docH, W, 3), 255, np.uint8)
    for x, ry, w, h, c in rects:
        img[int(ry):int(ry + h), int(x):int(x + w)] = c

    # union boxes + annotation records for every stamped element
    els = []
    stamped = ([body, *body.find_all(True)] if body is not None else [])
    stamped = [e for e in stamped if e.has_attr(STAMP_ATTR)]
    for el in stamped:
        sid = int(el[STAMP_ATTR])
        ids = [sid] + [int(d[STAMP_ATTR]) for d in el.find_all(True)
                       if d.has_attr(STAMP_ATTR)]
        ub = [own_box.get(i) for i in ids if own_box.get(i)]
        if ub:
            x1 = min(b[0] for b in ub); y1 = min(b[1] for b in ub)
            x2 = max(b[0] + b[2] for b in ub); y2 = max(b[1] + b[3] for b in ub)
            bx = (x1, y1, x2 - x1, y2 - y1)
        else:
            bx = (0, 0, 0, 0)
        st = _sty(el)
        bg = parse_css_color(st.get("background-color")
                             or st.get("background") or "")
        direct = " ".join("".join(
            str(c) for c in el.children
            if isinstance(c, NavigableString) and not isinstance(c, Comment)
        ).split())
        parent = el.parent
        pid = -1
        while parent is not None:
            if getattr(parent, "has_attr", None) and parent.has_attr(STAMP_ATTR):
                pid = int(parent[STAMP_ATTR]); break
            parent = parent.parent
        pad = _px(st.get("padding"))
        els.append({
            "id": sid, "tag": el.name.upper(), "parentId": pid,
            "x": bx[0], "y": bx[1], "w": bx[2], "h": bx[3],
            "display": "none" if st.get("display") == "none" else "block",
            "position": "static", "visibility": "visible", "opacity": 1.0,
            "paintable": sid not in hid_ids and bx[2] > 0 and bx[3] > 0,
            "displayNone": st.get("display") == "none",
            "leafTextLen": len(direct),
            "textLen": len(el.get_text(strip=True)),
            "bg": (f"rgb({int(bg[0])}, {int(bg[1])}, {int(bg[2])})"
                   if bg and bg[3] > 0.01 else "rgba(0, 0, 0, 0)"),
            "fontSize": _px(st.get("font-size")) or 16.0,
            "pad": [pad] * 4,
            "mar": [0.0, 0.0, 0.0, _px(st.get("margin-left"))],
            "borderW": [0.0] * 4,
            "transformed": False, "overflowVisible": True,
            "parentDisplay": "block",
            "nChildElems": len(el.find_all(True, recursive=False)),
            "nChildNodes": len(el.contents),
        })
    layout = {"boxes": boxes, "docW": W, "docH": docH,
              "text": "\n".join(texts)}
    ann = {"els": els, "docW": W, "docH": docH}
    return layout, img, ann


class FakePage:
    def __init__(self, outer):
        self._outer = outer

    def evaluate(self, _js):
        return self._outer._annotation


class FakeHarness:
    """Drop-in for RenderHarness in tests: same .render()/.page.evaluate()."""

    def __init__(self):
        self.page = FakePage(self)
        self._annotation = None
        self.n_renders = 0

    def render(self, html_path, out_png):
        html = Path(html_path).read_text(encoding="utf-8", errors="ignore")
        layout, img, ann = fake_engine(html)
        Path(out_png).parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(img).save(out_png)
        self._annotation = ann
        self.n_renders += 1
        return layout
