"""Level 0: deterministic rendering harness.

Contract: same HTML file -> same pixels, across runs. Both sides of every
comparison in this project are rendered by THIS harness, so blocked remote
assets and placeholder images affect original and compressed renders
identically -- comparisons stay valid.

Fixes over the old renderer:
  * viewport is FIXED (1280x800) and set once, before anything loads
  * full_page=True always (below-the-fold damage becomes detectable)
  * animations/transitions frozen, fonts awaited, scrollbars hidden
  * remote requests intercepted: images -> deterministic placeholder,
    everything else (remote CSS/JS/fonts) -> aborted
"""
from __future__ import annotations

import io
from pathlib import Path

from PIL import Image
from playwright.sync_api import sync_playwright

VIEWPORT = {"width": 1280, "height": 800}

FREEZE_CSS = (
    "*,*::before,*::after{animation:none!important;transition:none!important;"
    "caret-color:transparent!important;scroll-behavior:auto!important}"
)

# JS run inside the loaded page: visible INK boxes + doc size + visible text.
# Used by the acceptance gate (G2/G3/G4/G5).
#
# Why "ink" and not getBoundingClientRect():
#   A block with no background and no border paints nothing itself -- only its
#   text does. Its layout box can therefore change width dramatically with
#   ZERO change to the rendered pixels (unwrap a neutral <div> and a
#   left-aligned <p> may stretch from 100px to 1064px over empty space).
#   Measuring that box made G4 report center shifts of hundreds of pixels, and
#   made G5 average its CIEDE2000 over newly-included background, on renders
#   that were pixel-identical. In the 87-page WebCode2M stress run every safe
#   false-rejection was this artifact, and it set the zero-FPR bar so high
#   (center_shift 482px, deltae_max 10.5) that per-metric calibration
#   collapsed. Level 2 is DOM condensation, i.e. exactly this transformation,
#   so the gate has to measure what is painted, not what is laid out.
#
# Rule: an element contributes a box if it puts ink on the canvas -- it has
# direct text, is a replaced/atomic element, or paints a background, border or
# shadow. Its rect is where that ink actually lands: the border box for
# painted and replaced elements, the union of glyph rects for text-only ones.
EXTRACT_JS = """
() => {
  const ATOMIC = new Set(["IMG","INPUT","BUTTON","TEXTAREA","SELECT","SVG","VIDEO","HR"]);
  const px = v => parseFloat(v) || 0;

  const paints = cs => {
    const bg = cs.backgroundColor || "";
    const opaqueBg = bg !== "" && bg !== "transparent" &&
                     bg.indexOf("rgba(0, 0, 0, 0)") !== 0;
    const bgImg = (cs.backgroundImage || "none") !== "none";
    const border =
      (px(cs.borderTopWidth) > 0 && cs.borderTopStyle !== "none") ||
      (px(cs.borderRightWidth) > 0 && cs.borderRightStyle !== "none") ||
      (px(cs.borderBottomWidth) > 0 && cs.borderBottomStyle !== "none") ||
      (px(cs.borderLeftWidth) > 0 && cs.borderLeftStyle !== "none");
    return opaqueBg || bgImg || border || (cs.boxShadow || "none") !== "none";
  };

  const inkRect = (el, r) => {
    const range = document.createRange();
    range.selectNodeContents(el);
    const rects = range.getClientRects();
    let x1 = Infinity, y1 = Infinity, x2 = -Infinity, y2 = -Infinity;
    for (let i = 0; i < rects.length; i++) {
      const q = rects[i];
      if (q.width <= 0 || q.height <= 0) continue;
      if (q.left   < x1) x1 = q.left;
      if (q.top    < y1) y1 = q.top;
      if (q.right  > x2) x2 = q.right;
      if (q.bottom > y2) y2 = q.bottom;
    }
    if (x1 === Infinity) return r;
    return { x: x1, y: y1, width: x2 - x1, height: y2 - y1 };
  };

  const boxes = [];
  document.querySelectorAll("body *").forEach(el => {
    const cs = getComputedStyle(el);
    const r = el.getBoundingClientRect();
    const visible = r.width > 0 && r.height > 0 && cs.display !== "none" &&
                    cs.visibility !== "hidden" && parseFloat(cs.opacity) > 0.01;
    if (!visible) return;
    const leafText = [...el.childNodes].some(
        n => n.nodeType === 3 && n.textContent.trim().length > 0);
    const isAtomic = ATOMIC.has(el.tagName);
    const isPaint = paints(cs);
    if (!(leafText || isAtomic || isPaint)) return;
    const k = (isAtomic || isPaint) ? r : inkRect(el, r);
    if (k.width <= 0 || k.height <= 0) return;
    boxes.push({
      text: (el.innerText || "").replace(/\\s+/g, " ").trim().slice(0, 120),
      tag: el.tagName,
      kind: isAtomic ? "atomic" : (isPaint ? "paint" : "text"),
      x: k.x + scrollX, y: k.y + scrollY, w: k.width, h: k.height
    });
  });
  return {
    boxes: boxes,
    docW: document.documentElement.scrollWidth,
    docH: document.documentElement.scrollHeight,
    text: (document.body ? document.body.innerText : "")
  };
}
"""


def _placeholder_png_bytes() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (2, 2), (204, 204, 204)).save(buf, format="PNG")
    return buf.getvalue()


class RenderHarness:
    """Reusable deterministic renderer. Create once, use for many pages.

    Usage:
        with RenderHarness() as h:
            h.load("page.html")
            layout = h.extract_layout()
            h.screenshot("page.png")
    """

    def __init__(self, headless: bool = True):
        self._placeholder = _placeholder_png_bytes()
        self._pw = sync_playwright().start()
        self.browser = self._pw.chromium.launch(
            headless=headless,
            args=[
                "--force-color-profile=srgb",
                "--disable-lcd-text",
                "--hide-scrollbars",
                "--font-render-hinting=none",
                "--force-device-scale-factor=1",
            ],
        )
        self.ctx = self.browser.new_context(
            viewport=VIEWPORT,
            device_scale_factor=1,
            color_scheme="light",
            locale="en-US",
            timezone_id="UTC",
            reduced_motion="reduce",
        )
        self.page = self.ctx.new_page()
        self.page.route("**/*", self._intercept)

    # -- request interception -------------------------------------------------
    def _intercept(self, route):
        url = route.request.url
        if url.startswith(("http://", "https://")):
            if route.request.resource_type == "image":
                route.fulfill(status=200, content_type="image/png",
                              body=self._placeholder)
            else:
                route.abort()
        else:  # file:// and data: URIs are allowed
            route.continue_()

    # -- core operations ------------------------------------------------------
    def load(self, html_path) -> None:
        uri = Path(html_path).resolve().as_uri()
        self.page.goto(uri, wait_until="load", timeout=30000)
        self.page.add_style_tag(content=FREEZE_CSS)
        try:
            self.page.evaluate("() => document.fonts ? document.fonts.ready : true")
        except Exception:
            pass
        self.page.wait_for_timeout(200)

    def extract_layout(self) -> dict:
        """Return {boxes, docW, docH, text} for the currently loaded page."""
        return self.page.evaluate(EXTRACT_JS)

    def screenshot(self, out_png) -> None:
        Path(out_png).parent.mkdir(parents=True, exist_ok=True)
        self.page.screenshot(path=str(out_png), full_page=True)  # FULL PAGE, always

    def render(self, html_path, out_png) -> dict:
        """Convenience: load + extract + screenshot. Returns the layout dict."""
        self.load(html_path)
        layout = self.extract_layout()
        self.screenshot(out_png)
        return layout

    # -- lifecycle ------------------------------------------------------------
    def close(self) -> None:
        try:
            self.browser.close()
        finally:
            self._pw.stop()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()