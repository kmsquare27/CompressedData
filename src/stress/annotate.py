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
    """True if inline CSS could style THROUGH THE STAMP ATTRIBUTE, in which
    case stamping ids could itself change rendering. CSS attribute selectors
    name attributes exactly (there is no wildcard on the attribute NAME), so
    only a selector containing `data-mut-id` is a hazard. The previous check
    (`"[data-" in css`) skipped every page that styles any data-* attribute
    of its own -- 2 pages and 44 stress samples for nothing. Remote CSS is
    aborted by the harness, so <style> blocks are the only live CSS source."""
    soup = BeautifulSoup(html, "lxml")
    css = " ".join(s.get_text() or "" for s in soup.find_all("style"))
    return STAMP_ATTR in css


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
  const SVG_NS = "http://www.w3.org/2000/svg";
  // Inherited properties: a wrapper whose value differs from its parent's is
  // load-bearing for its children even with zero box/background (e.g. a
  // text-align:center or color wrapper). The old neutral() predicate never
  // checked these, which is one reason 66% of collapse candidates were
  // blacklisted by the render gate.
  const INHERITED = ["color","font-family","font-size","font-weight","font-style",
    "font-variant","font-stretch","line-height","letter-spacing","word-spacing",
    "text-align","text-align-last","text-indent","text-transform","white-space",
    "word-break","overflow-wrap","direction","visibility","list-style-type",
    "list-style-position","list-style-image","quotes","tab-size","hyphens",
    "text-shadow","writing-mode","text-orientation","caption-side","border-collapse",
    "border-spacing","empty-cells","-webkit-text-fill-color","-webkit-text-stroke-width",
    "-webkit-text-stroke-color","font-feature-settings","font-kerning",
    "font-variant-ligatures","font-variant-numeric","text-rendering",
    "text-underline-position","text-underline-offset","text-decoration-thickness",
    "image-rendering","color-scheme","line-break","orphans","widows"];

  // `clip: rect(...)` with an empty rect (applies to absolutely positioned
  // elements only) -- the .sr-only / .visually-hidden pattern.
  const clipZero = cs => {
    if (cs.position !== "absolute" && cs.position !== "fixed") return false;
    const m = /rect\\(\\s*([-\\d.]+)px,?\\s*([-\\d.]+)px,?\\s*([-\\d.]+)px,?\\s*([-\\d.]+)px\\s*\\)/
              .exec(cs.clip || "");
    if (!m) return false;
    const t = +m[1], r = +m[2], b = +m[3], l = +m[4];
    return (r - l) <= 0 || (b - t) <= 0;
  };
  // clip-path: inset(50%) or more clips everything.
  const clipPathZero = cs => {
    const m = /^inset\\(\\s*([\\d.]+)%\\s*\\)$/.exec(cs.clipPath || "");
    return !!m && (+m[1]) >= 50;
  };
  // Border box entirely outside the clip region of an overflow!=visible
  // ancestor (e.g. text inside a 1x1 overflow:hidden box, or an off-canvas
  // menu inside an overflow:hidden container). CSS2.1 11.1.1: an ancestor
  // clips a descendant unless the descendant's containing block is above
  // the ancestor; approximated conservatively -- absolutely positioned boxes
  // are only clipped by POSITIONED ancestors, fixed boxes never.
  const clippedAway = (el, cs, r) => {
    if (cs.position === "fixed") return false;
    let x1 = r.left, y1 = r.top, x2 = r.right, y2 = r.bottom;
    if (x2 - x1 <= 0 || y2 - y1 <= 0) return false;
    for (let a = el.parentElement; a && a !== document.documentElement; a = a.parentElement) {
      const acs = getComputedStyle(a);
      const clips = acs.overflowX !== "visible" || acs.overflowY !== "visible" ||
                    (acs.contain || "").indexOf("paint") >= 0;
      if (!clips) continue;
      if (cs.position === "absolute" && acs.position === "static" &&
          acs.transform === "none" && acs.filter === "none") continue;
      const ar = a.getBoundingClientRect();
      x1 = Math.max(x1, ar.left + px(acs.borderLeftWidth));
      y1 = Math.max(y1, ar.top + px(acs.borderTopWidth));
      x2 = Math.min(x2, ar.right - px(acs.borderRightWidth));
      y2 = Math.min(y2, ar.bottom - px(acs.borderBottomWidth));
      if (x2 - x1 <= 0 || y2 - y1 <= 0) return true;
    }
    return false;
  };

  const els = [];
  document.querySelectorAll("[data-mut-id]").forEach(el => {
    const cs = getComputedStyle(el);
    const r = el.getBoundingClientRect();
    let leafLen = 0;
    el.childNodes.forEach(n => {
      if (n.nodeType === 3) leafLen += n.textContent.trim().length;
    });
    const parent = el.parentElement;
    const pcs = parent ? getComputedStyle(parent) : null;
    const zeroArea = !(r.width > 0 && r.height > 0);
    const offCanvas = r.right <= 0 || r.bottom <= 0;
    const hidden = cs.display === "none" || cs.visibility === "hidden" ||
                   px(cs.opacity) <= 0.01;
    const clipped = !zeroArea && !hidden &&
                    (clipZero(cs) || clipPathZero(cs) || clippedAway(el, cs, r));
    const inSvg = el.namespaceURI === SVG_NS && el.tagName.toLowerCase() !== "svg";
    const inheritDiff = !!pcs && INHERITED.some(
        p => cs.getPropertyValue(p) !== pcs.getPropertyValue(p));
    const wrapperHazard = inheritDiff ||
        cs.float !== "none" || cs.clear !== "none" ||
        (cs.textDecorationLine || "none") !== "none" ||
        (cs.boxShadow || "none") !== "none" ||
        (px(cs.outlineWidth) > 0 && cs.outlineStyle !== "none") ||
        (cs.columnCount || "auto") !== "auto" ||
        (cs.contain || "none") !== "none" || cs.isolation === "isolate" ||
        (cs.mixBlendMode || "normal") !== "normal" ||
        (cs.clipPath || "none") !== "none" || (cs.maskImage || "none") !== "none" ||
        (cs.minHeight !== "0px" && cs.minHeight !== "auto") ||
        (cs.minWidth !== "0px" && cs.minWidth !== "auto") ||
        cs.maxHeight !== "none" || cs.maxWidth !== "none" ||
        (cs.display === "inline" && cs.verticalAlign !== "baseline") ||
        (pcs && pcs.display.startsWith("table") && pcs.display !== "table-cell");
    els.push({
      id: parseInt(el.getAttribute("data-mut-id"), 10),
      tag: el.tagName,
      parentId: (parent && parent.hasAttribute("data-mut-id"))
                ? parseInt(parent.getAttribute("data-mut-id"), 10) : -1,
      x: r.x + scrollX, y: r.y + scrollY, w: r.width, h: r.height,
      display: cs.display, position: cs.position,
      visibility: cs.visibility, opacity: px(cs.opacity),
      paintable: !zeroArea && !offCanvas && !hidden && !clipped,
      clipped: clipped, inSvg: inSvg,
      wrapperHazard: !!wrapperHazard, inheritDiff: inheritDiff,
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
# In-page PRESCREEN of Level-2 edit candidates (no renders)
# ---------------------------------------------------------------------------
# For each candidate edit independently: apply it to the live DOM, ask the
# oracle whether any remaining element's rect/computed style changed, then
# restore the DOM exactly. Removal is simulated by swapping the element for
# a comment node (leaves neighbouring whitespace text nodes in place, exactly
# as BeautifulSoup's decompose() does); collapse by moving the children out.
# An edit passes only if every changed/missing key belongs to the removed
# subtree (or is the collapsed wrapper itself). The margin-collapse trap and
# descendant-selector breakage (`.wrapper p {...}`) both show up here as
# rect/style diffs on OTHER elements, which is why ~2/3 of collapse
# candidates used to be blacklisted only after burning render budget.
# Restore is verified: if the DOM does not come back identical, the page's
# prescreen is aborted and the runner falls back to unfiltered gating.
PRESCREEN_JS = """
(edits) => {
  const O = window.__oracle;
  if (!O) return null;
  const byId = new Map();
  document.querySelectorAll("[data-mut-id]").forEach(
      el => byId.set(parseInt(el.getAttribute("data-mut-id"), 10), el));
  const out = [];
  for (const [kind, id] of edits) {
    const el = byId.get(id);
    if (!el || !el.parentNode) { out.push({kind, id, ok: false, reason: "not found"}); continue; }
    const expected = new Set(["m" + id]);
    let restore = null;
    if (kind === "remove") {
      el.querySelectorAll("[data-mut-id]").forEach(
          d => expected.add("m" + d.getAttribute("data-mut-id")));
      const ph = document.createComment("mut");
      el.replaceWith(ph);
      restore = () => ph.replaceWith(el);
    } else if (kind === "collapse") {
      const kids = Array.from(el.childNodes);
      if (!kids.length) { out.push({kind, id, ok: false, reason: "no children"}); continue; }
      el.replaceWith(...kids);
      restore = () => { kids[0].replaceWith(el); el.append(...kids); };
    } else { out.push({kind, id, ok: false, reason: "unknown kind"}); continue; }
    let res;
    try { res = O.compare(); } catch (e) { res = {nDiff: -1, first: "compare error: " + e, missing: [], added: []}; }
    restore();
    const chk = O.compare();
    const restored = chk.nDiff === 0 && chk.missing.length === 0 && chk.added.length === 0;
    const extraMissing = res.missing.filter(k => !expected.has(k));
    out.push({kind, id,
              ok: res.nDiff === 0 && res.added.length === 0 && extraMissing.length === 0,
              nDiff: res.nDiff, first: res.first, extraMissing: extraMissing.length,
              restored});
    if (!restored) { out.push({kind: "abort", id: -1, ok: false, reason: "restore failed"}); break; }
  }
  return out;
}
"""


def prescreen_edits(harness, edits) -> list | None:
    """Prescreen [(kind, id), ...] on the CURRENTLY LOADED stamped page.
    Returns one dict per edit ({kind, id, ok, nDiff, first, ...}) in order,
    or None if the oracle could not run / restore failed part-way (the
    caller should then gate the unfiltered list)."""
    try:
        harness.oracle_baseline()
        res = harness.page.evaluate(PRESCREEN_JS, [list(e) for e in edits])
    except Exception as e:  # noqa: BLE001
        print(f"[annotate] prescreen unavailable: {e}")
        return None
    if not res or any(r.get("kind") == "abort" for r in res):
        return None
    return res


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


def ssim_rgb_lowmem(a: np.ndarray, b: np.ndarray):
    """Delegates to the gate's implementation so the stress test's SSIM and
    the gate's diagnostic SSIM are literally the same code (same-code-path
    principle -- the number you calibrate on is the number you report).
    Never raises: SSIM is diagnostic, and a null column must not cost a
    multi-hour run."""
    try:
        from src.compare.gate import _ssim_rgb_lowmem
        return _ssim_rgb_lowmem(a, b)
    except Exception:
        return None


def ssim_union_canvas(a: np.ndarray, b: np.ndarray):
    """SSIM with both renders padded (white) onto the union canvas, so
    missing/added content COUNTS AGAINST the score instead of being cropped
    away. Computed alongside the gate's common-crop ssim_diag; the SSIM
    kill-shot analysis uses the MORE charitable of the two per sample, so
    'SSIM alone is insufficient' cannot be attributed to an unfair SSIM
    implementation."""
    H = max(a.shape[0], b.shape[0])
    W = max(a.shape[1], b.shape[1])

    def pad(x):
        if x.shape[0] == H and x.shape[1] == W:
            return x[:, :, :3]
        c = np.full((H, W, 3), 255, dtype=np.uint8)
        c[:x.shape[0], :x.shape[1]] = x[:, :, :3]
        return c

    try:
        pa, pb = pad(a), pad(b)
    except Exception:
        return None
    v = ssim_rgb_lowmem(pa, pb)
    del pa, pb
    return v