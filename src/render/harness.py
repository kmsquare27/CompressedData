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

# JS run inside the loaded page: visible atomic boxes + doc size + visible text.
# Used by the acceptance gate (G2/G3/G4/G5).
EXTRACT_JS = """
() => {
  const ATOMIC = new Set(["IMG","INPUT","BUTTON","TEXTAREA","SELECT","SVG","VIDEO","HR"]);
  const boxes = [];
  document.querySelectorAll("body *").forEach(el => {
    const cs = getComputedStyle(el);
    const r = el.getBoundingClientRect();
    const visible = r.width > 0 && r.height > 0 && cs.display !== "none" &&
                    cs.visibility !== "hidden" && parseFloat(cs.opacity) > 0.01;
    if (!visible) return;
    const leafText = [...el.childNodes].some(
        n => n.nodeType === 3 && n.textContent.trim().length > 0);
    if (leafText || ATOMIC.has(el.tagName)) {
      boxes.push({
        text: (el.innerText || "").replace(/\\s+/g, " ").trim().slice(0, 120),
        tag: el.tagName,
        x: r.x + scrollX, y: r.y + scrollY, w: r.width, h: r.height
      });
    }
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
     try:
        self.page.goto(uri, wait_until="load", timeout=60000)
     except Exception:                                   # slow page: fall back
        self.page.goto(uri, wait_until="domcontentloaded", timeout=60000)
     self.page.add_style_tag(content=FREEZE_CSS)
     try:
        self.page.evaluate("() => document.fonts ? document.fonts.ready : true")
     except Exception:
        pass
     self.page.wait_for_timeout(400)                     # was 200

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
