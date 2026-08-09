"""Level 1: safe structural + syntactic minification (upgraded).

Changes vs the old level1_minify:
  * whitespace handled by `minify-html` (Rust, spec-aware) instead of regex --
    the old `re.sub(r">\\s+<", "><")` could merge visually-significant spaces
    between inline elements; minify-html knows the inline rules and protects
    <pre>/<textarea>. CSS inside <style> is syntactically minified for free.
  * data-* removal is CONDITIONAL: skipped when any stylesheet on the page
    contains "[data-" (attribute selectors can style through data attributes).
  * all aria-* removed (they never affect pixels; note the accessibility
    limitation once in the paper), plus title="" attributes.
  * <meta> pruned except charset and viewport (viewport affects rendering!),
    <link> pruned unless rel contains "stylesheet".
Every page still goes through the acceptance gate afterwards -- these rules
are "expected safe", the gate is the proof.
"""
from __future__ import annotations

import re

from bs4 import BeautifulSoup, Comment


def level1_minify(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")

    # Is any data-* attribute referenced by CSS attribute selectors?
    style_text = " ".join(s.get_text() for s in soup.find_all("style"))
    data_attr_styled = "[data-" in style_text

    # Comments
    for c in soup.find_all(string=lambda t: isinstance(t, Comment)):
        c.extract()

    # Script-like elements
    for t in soup.find_all(["script", "noscript"]):
        t.decompose()

    # <link>: keep stylesheets only
    for t in soup.find_all("link"):
        rel = t.get("rel", [])
        rel = " ".join(rel).lower() if isinstance(rel, list) else str(rel).lower()
        if "stylesheet" not in rel:
            t.decompose()

    # <meta>: keep charset + viewport only (viewport affects rendering)
    for t in soup.find_all("meta"):
        keep = t.has_attr("charset") or str(t.get("name", "")).lower() == "viewport"
        if not keep:
            t.decompose()

    # Attribute pruning
    for t in soup.find_all(True):
        for attr in list(t.attrs):
            al = attr.lower()
            if al.startswith("on"):
                del t.attrs[attr]
            elif al.startswith("aria-"):
                del t.attrs[attr]
            elif al == "title":
                del t.attrs[attr]
            elif al.startswith("data-") and not data_attr_styled:
                del t.attrs[attr]

    out = str(soup)

    # Spec-aware syntactic minification
    try:
        import minify_html
        try:
            out = minify_html.minify(out, minify_css=True, keep_closing_tags=True,
                                     do_not_minify_doctype=True)
        except TypeError:  # older/newer API surface
            out = minify_html.minify(out)
    except Exception:
        # Conservative fallback only: collapse runs of whitespace to one space.
        # (HTML collapses runs anyway outside <pre>; do NOT strip between tags.)
        out = re.sub(r"[ \t\r\n]{2,}", " ", out)

    return out.strip()
