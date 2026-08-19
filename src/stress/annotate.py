"""Annotation + ground-truth verification for the mutation stress test.

Two jobs:

1) ANNOTATE-THEN-EDIT. Mutation operators must target elements that are
   really visible and must edit RELATIVE to real computed values
   (font-size+2px, margin+16px, background shifted to a target CIEDE2000).
   BeautifulSoup alone cannot know any of that. So: stamp every body element
   with data-mut-id offline, render the stamped page once under the
   deterministic harness, and collect per-element browser facts (paintable
   rect under the DOCUMENT-BOUNDS rule -- below the fold IS visible --
   computed styles, text presence, parent linkage). Operators then pick
   targets from verified candidates. This mirrors the pipeline's own
   Level-2 annotate-in-browser -> prune-offline design.

2) VERIFY EVERY LABEL. "Breaking" and "safe" are *intent* labels; the
   calibration needs *ground-truth* labels. Because step 01 proves the
   renderer deterministic (same HTML -> same pixels), any pixel difference
   between the stamped original's render and a mutant's render is caused by
   the mutation. So:
     breaking mutant, zero pixel diff  -> a dud (the operator hit an
        invisible/no-op target); the runner resamples another target, and
        if it keeps landing on duds the sample is EXCLUDED, never counted
        as a gate miss.
     safe mutant, nonzero pixel diff   -> the "safe" edit was not safe on
        this page (e.g. whitespace between inline elements). EXCLUDED from
        the safe calibration set and reported as a finding -- this is
        Stage 4.3's "safety is not decidable statically" point, measured.
   Without this verification, mislabeled mutants poison recall and
   false-rejection estimates in both directions.

The stamped file is the comparison baseline for all of its mutants, so the
data-mut-id attributes appear on BOTH sides of every comparison and cancel
out. data-* attributes do not render -- unless page CSS contains [data-
attribute selectors, which data_attr_selector_hazard() screens for (same
guard Level 1 uses); such pages are skipped and counted.
"""
from __future__ import annotations

import numpy as np
from bs4 import BeautifulSoup

STAMP_ATTR = "data-mut-id"

# ---------------------------------------------------------------------------
# Stamping (offline, BeautifulSoup)
# ---------------------------------------------------------------------------


def data_attr_selector_hazard(html: str) -> bool:
    """True if inline CSS could style via data attributes, in which case
    stamping ids could itself change rendering. Remote CSS is aborted by the
    harness, so <style> blocks are the only live CSS source to check."""
    soup = BeautifulSoup(html, "lxml")
    css = " ".join(s.get_text() or "" for s in soup.find_all("style"))
    return "[data-" in css


def stamp_ids(html: str):
    """Stamp data-mut-id on <body> and every element under it, in document
    order. Returns (stamped_html, n_stamped) or (None, 0) if no <body>."""
    soup = BeautifulSoup(html, "lxml")
    body = soup.body
    if body is None:
        return None, 0
    n = 0
    for el in [body, *body.find_all(True)]:
        el[STAMP_ATTR] = str(n)
        n += 1
    return str(soup), n


# ---------------------------------------------------------------------------
# Browser-side metadata (run on the loaded, stamped page)
# ---------------------------------------------------------------------------
# paintable follows the protocol's document-bounds rule for full-page
# capture: zero-area or parked off-canvas (fully above/left) is invisible;
# BELOW THE FOLD IS VISIBLE. display:none / visibility:hidden / opacity<=0.01
# also make a subtree unpaintable.

ANNOTATE_JS = """
() => {
  const px = v => parseFloat(v) || 0;
  const els = [];
  document.querySelectorAll("[data-mut-id]").forEach(el => {
    const cs = getComputedStyle(el);
    const r = el.getBoundingClientRect();
    let leafLen = 0;
    el.childNodes.forEach(n => {
      if (n.nodeType === 3) leafLen += n.textContent.trim().length;
    });
    const parent = el.parentElement;
    els.push({
      id: parseInt(el.getAttribute("data-mut-id"), 10),
      tag: el.tagName,
      parentId: (parent && parent.hasAttribute("data-mut-id"))
                ? parseInt(parent.getAttribute("data-mut-id"), 10) : -1,
      x: r.x + scrollX, y: r.y + scrollY, w: r.width, h: r.height,
      display: cs.display, position: cs.position,
      visibility: cs.visibility, opacity: px(cs.opacity),
      paintable: r.width > 0 && r.height > 0 && r.right > 0 && r.bottom > 0 &&
                 cs.display !== "none" && cs.visibility !== "hidden" &&
                 px(cs.opacity) > 0.01,
      displayNone: cs.display === "none",
      leafTextLen: leafLen,
      textLen: (el.textContent || "").trim().length,
      bg: cs.backgroundColor,
      fontSize: px(cs.fontSize),
      pad: [px(cs.paddingTop), px(cs.paddingRight),
            px(cs.paddingBottom), px(cs.paddingLeft)],
      mar: [px(cs.marginTop), px(cs.marginRight),
            px(cs.marginBottom), px(cs.marginLeft)],
      borderW: [px(cs.borderTopWidth), px(cs.borderRightWidth),
                px(cs.borderBottomWidth), px(cs.borderLeftWidth)],
      transformed: cs.transform !== "none" || cs.filter !== "none",
      overflowVisible: cs.overflowX === "visible" && cs.overflowY === "visible",
      parentDisplay: parent ? getComputedStyle(parent).display : "block",
      nChildElems: el.children.length,
      nChildNodes: el.childNodes.length
    });
  });
  return { els: els,
           docW: document.documentElement.scrollWidth,
           docH: document.documentElement.scrollHeight };
}
"""


class PageMeta:
    """Per-element browser facts for one stamped page, keyed by data-mut-id."""

    def __init__(self, raw: dict):
        self.docW = raw.get("docW", 0)
        self.docH = raw.get("docH", 0)
        self.els = {e["id"]: e for e in raw.get("els", [])}

    def in_doc_order(self):
        return [self.els[i] for i in sorted(self.els)]

    def candidates(self, pred, exclude=()):
        """Elements passing pred, in document order (deterministic)."""
        ex = set(exclude)
        return [e for e in self.in_doc_order() if e["id"] not in ex and pred(e)]


def collect_meta(harness) -> PageMeta:
    """Collect metadata from the page currently loaded in the harness.
    Call immediately after render_page() on the stamped original, while it
    is still the loaded page."""
    return PageMeta(harness.page.evaluate(ANNOTATE_JS))


# ---------------------------------------------------------------------------
# Ground-truth verification (pixel diff under the deterministic harness)
# ---------------------------------------------------------------------------


def pixel_stats(a: np.ndarray, b: np.ndarray) -> dict:
    """Pixel-level difference between two full-page renders. NEVER resizes:
    every pixel outside the common area counts as different, so a page that
    lost its lower half scores as massively changed (a resize would hide
    exactly that failure -- protocol Bug 3)."""
    ha, wa = a.shape[:2]
    hb, wb = b.shape[:2]
    h, w = min(ha, hb), min(wa, wb)
    diff = np.any(a[:h, :w, :3] != b[:h, :w, :3], axis=-1)
    n_common = int(diff.sum())
    n_outside = ha * wa + hb * wb - 2 * h * w
    n_diff = n_common + n_outside
    denom = max(ha, hb) * max(wa, wb)
    out = {
        "n_diff_pixels": int(n_diff),
        "diff_frac": round(n_diff / max(denom, 1), 6),
        "shape_mismatch": (ha, wa) != (hb, wb),
        "diff_bbox": "",
    }
    if n_common:
        ys, xs = np.where(diff)
        out["diff_bbox"] = f"{int(xs.min())},{int(ys.min())},{int(xs.max())},{int(ys.max())}"
    return out


def visually_changed(stats: dict) -> bool:
    return stats["n_diff_pixels"] > 0


def ssim_union_canvas(a: np.ndarray, b: np.ndarray):
    """SSIM with both renders padded (white) onto the union canvas, so
    missing/added content COUNTS AGAINST the score instead of being cropped
    away. Computed alongside the gate's common-crop ssim_diag; the SSIM
    kill-shot analysis uses the MORE charitable of the two per sample, so
    'SSIM alone is insufficient' cannot be attributed to an unfair SSIM
    implementation."""
    try:
        from skimage.metrics import structural_similarity as ssim
    except Exception:
        return None
    H = max(a.shape[0], b.shape[0])
    W = max(a.shape[1], b.shape[1])

    def pad(x):
        c = np.full((H, W, 3), 255, dtype=np.uint8)
        c[:x.shape[0], :x.shape[1]] = x[:, :, :3]
        return c

    return round(float(ssim(pad(a), pad(b), channel_axis=2, data_range=255)), 5)
