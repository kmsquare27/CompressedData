"""Level 1, split into two stages (v3).

WHY THE SPLIT. The old level1_minify handed the whole document to minify-html.
Two things were wrong with that:

  1. minify-html deletes ALL whitespace between the children of "layout" tags
     (div, body, ul, ...) even when those children are inline or inline-block
     -- by design ("useful when using display:inline-block so that whitespace
     between elements does not alter layout", its README). So
     <div><a>Home</a> <a>About</a></div> becomes HomeAbout, and two buttons in
     a div lose their ~4px gap. That is the root cause of the 33 Level-1 gate
     rejections (median centre shift 4.2px = one space at 14-16px) and of the
     26 "unsafe S6" stress findings. It is a policy of one tool, not a
     property of minification.
  2. The call passed `do_not_minify_doctype`, which minify-html renamed to
     `minify_doctype` in 0.16. On a current install the TypeError fallback
     ran `minify_html.minify(out)` with ALL defaults: minify_css=False (no CSS
     minification at all) and keep_closing_tags=False (optional closing tags
     and <html>/<head> silently dropped). Whether the report's Level-1 numbers
     came from that branch is unknown until checked; this file makes it
     impossible to happen again: a wrong keyword argument raises.

STAGES.
  L1a  proposes pruning and CSS-minification transformations; the gate
       checks their rendered output. Counterexamples exist for every
       operator (e.g. [aria-hidden] selectors, content: attr(title),
       DOM-writing inline scripts). Never touches HTML whitespace.
         - HTML comments, <script>, <noscript>
         - on*, aria-*, title, data-* (unless a [data- selector exists)
         - <meta> except charset/viewport; <link> except stylesheets
         - CSS-only syntactic minification of every <style> block and every
           style="" attribute through lightningcss (minify-html's CSS engine,
           reached by wrapping the bare stylesheet in <style>...</style> so
           the HTML itself is never handed to minify-html).
       lightningcss already does everything a "baseline CSS compression"
       level would: comments, whitespace, .5px, #fff/red, four-value and
       longhand->shorthand collapsing, duplicate declarations, and merging
       of ADJACENT rules with identical declarations. Reimplementing it with
       tinycss2 would yield ~0 tokens.
  L1b  gated HTML-syntax pass: minify-html on the document with CSS
       minification OFF (done in L1a), closing tags and <html>/<head> kept so
       the training dialect stays standard HTML. Its whitespace policy is the
       one described above, so L1b rejections are expected and are reported
       separately by 02_run_level1.py.

Every stage still goes through the acceptance gate; "by construction" is the
expectation, the gate is the proof.
"""
from __future__ import annotations

import re

from bs4 import BeautifulSoup, Comment
from bs4.element import Stylesheet

# Spec-compliant WHATWG parsing -- the same algorithm Chromium uses, so a
# parse+serialize round trip re-parses to an identical DOM. Measured on 15
# malformed WebCode2M pages: a no-op round trip changed the render on 14/15
# under lxml but 1/15 under html5lib. Every BeautifulSoup call in this project
# must use this parser; if stamp_ids() and apply_edits() disagree, Level 2
# re-introduces the damage between stamping and editing.
HTML_PARSER = "html5lib"


def style_text(tag) -> str:
    """CSS text of a <style> element, independent of the parser.

    NOT interchangeable with tag.get_text(). A <style> tag's
    interesting_string_types is Stylesheet, and only bs4's lxml builder wraps
    stylesheet text in a Stylesheet node -- the html5lib builder emits a plain
    NavigableString, which get_text() then filters out and returns "". Reading
    .contents is faithful under both (verified: no entity substitution, so CSS
    child selectors like `a > b` survive). Every "[data-" guard depends on
    this; with get_text() under html5lib they would all silently see no CSS.
    """
    return "".join(str(c) for c in tag.contents)


_STYLE_WRAP = re.compile(r"^<style>(.*)</style>$", re.S)


# ---------------------------------------------------------------------------
# CSS-only minification (lightningcss via minify-html, HTML never involved)
# ---------------------------------------------------------------------------
def minify_css_text(css: str) -> str:
    """lightningcss minification of a bare stylesheet.

    Returns the input UNCHANGED on any failure, on a parse error (minify-html
    then returns the CSS as-is, verified), or when the result is not shorter.
    A wrong keyword argument raises: no silent fallback."""
    if not css or not css.strip():
        return css
    import minify_html  # ImportError is a hard failure, on purpose
    out = minify_html.minify("<style>" + css + "</style>", minify_css=True)
    m = _STYLE_WRAP.match(out)
    if not m:
        return css
    new = m.group(1)
    return new if len(new) < len(css) else css


def minify_style_attr(value: str) -> str:
    """Minify a style="" attribute value through the same engine by wrapping
    it in a dummy rule. Returns the input unchanged on any failure."""
    if not value or not value.strip():
        return value
    out = minify_css_text("x{" + value + "}")
    if out.startswith("x{") and out.endswith("}"):
        inner = out[2:-1]
        return inner if len(inner) < len(value) else value
    return value


def _introduces_non_ascii(old: str, new: str) -> bool:
    """lightningcss rewrites escapes like \\201C to the literal character.
    That is only safe if the document declares its encoding; otherwise the
    browser may sniff the file:// document as windows-1252 and render
    mojibake -- a real, gate-detectable render change, avoided up front."""
    return any(ord(c) > 127 for c in new) and not any(ord(c) > 127 for c in old)


def _charset_declared(soup: BeautifulSoup) -> bool:
    if soup.find("meta", charset=True):
        return True
    for m in soup.find_all("meta"):
        if str(m.get("http-equiv", "")).lower() == "content-type" \
                and "charset" in str(m.get("content", "")).lower():
            return True
    return False


# ---------------------------------------------------------------------------
# L1a -- render-neutral by construction
# ---------------------------------------------------------------------------
def level1a(html: str) -> str:
    soup = BeautifulSoup(html, HTML_PARSER)

    # Is any data-* attribute referenced by CSS attribute selectors? (This is
    # the page's OWN data-* attributes, so the broad "[data-" check is right
    # here -- unlike the stamping hazard in stress/annotate.py.)
    css_all = " ".join(style_text(s) for s in soup.find_all("style"))
    data_attr_styled = "[data-" in css_all
    charset_ok = _charset_declared(soup)

    # Comments (HTML only; CSS comments are handled by lightningcss below)
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
        keep = t.has_attr("charset") or str(t.get("name", "")).lower() == "viewport" \
            or str(t.get("http-equiv", "")).lower() == "content-type"
        if not keep:
            t.decompose()

    # Attribute pruning + style="" minification
    for t in soup.find_all(True):
        for attr in list(t.attrs):
            al = attr.lower()
            if al.startswith("on") or al.startswith("aria-") or al == "title":
                del t.attrs[attr]
            elif al.startswith("data-") and not data_attr_styled:
                del t.attrs[attr]
        if t.has_attr("style") and isinstance(t["style"], str):
            new = minify_style_attr(t["style"])
            if new != t["style"] and (charset_ok
                                      or not _introduces_non_ascii(t["style"], new)):
                t["style"] = new

    # <style> blocks: CSS-only minification, HTML untouched
    for s in soup.find_all("style"):
        css = s.string if s.string is not None else style_text(s)
        if not css:
            continue
        new = minify_css_text(css)
        if new != css and (charset_ok or not _introduces_non_ascii(css, new)):
            s.clear()
            s.append(Stylesheet(new))   # Stylesheet: no entity substitution

    return str(soup).strip()


# ---------------------------------------------------------------------------
# L1b -- gated HTML-syntax pass (whitespace, attribute quotes, ...)
# ---------------------------------------------------------------------------
def level1b(html: str) -> str:
    """minify-html over the whole document. CSS minification OFF (L1a did it
    without touching HTML), closing tags and <html>/<head> KEPT so the
    training dialect stays standard. Requires minify-html >= 0.16; an older
    version raises TypeError on `minify_doctype`, which is the intended
    behaviour (pin the dependency, do not fall back silently)."""
    import minify_html
    return minify_html.minify(
        html,
        minify_css=False,
        minify_js=False,
        keep_closing_tags=True,
        keep_html_and_head_opening_tags=True,
        minify_doctype=False,
    ).strip()


# Backward-compatible name: "Level 1" now means the by-construction stage.
def level1_minify(html: str) -> str:
    return level1a(html)
