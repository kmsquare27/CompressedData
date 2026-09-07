"""Level 3: invisible CSS under the training render.

Level 2's finding was that what compresses on real pages is DOM content that
never renders. Level 3 is the same claim for CSS: rules that never contribute
to any computed style in the TRAINING STATE (fixed 1280px viewport, static,
no pointer/keyboard, animations frozen, remote assets blocked -- exactly what
the harness renders and the model is trained on). All operators are
subtractive and are EXPECTED not to change any computed style; the in-page
oracle checks that expectation and the runner's acceptance criterion is
PIXEL IDENTITY, not the tolerance gate, because a rule can influence the
cascade without styling a matched element (@layer order is one example). That matters:
G4 measures boxes and G5 a per-block MEAN colour, so neither would notice a
dropped border-radius, text-decoration or box-shadow; the mutation benchmark
never contained those operators, and Level 3 does not rely on it.

Operators (each is a bucket in the accounting):
  unmatched       style rule whose every selector matches no element
  stateful        selector gated on :hover/:active/:focus(-within/-visible)/
                  :visited/:target AND matching nothing in the captured state
                  (an <input autofocus> makes :focus true; the browser is
                  asked, the pseudo-class alone decides only the bucket label)
  selector_trim   drop the dead items of a selector LIST, keep the live ones
  media_nomatch   @media whose query is false at the harness (matchMedia)
  supports_nomatch @supports whose condition is false (CSS.supports)
  block_emptied   conditional block whose every inner rule was dead
  keyframes       @keyframes (animations are frozen by FREEZE_CSS)
  font_face_remote @font-face whose every source is unreachable under the
                  harness (http(s) aborted; relative paths checked on disk;
                  local()/data: keep)
  import          @import whose target is unreachable, same rule
  page            @page (print only)
  dead_props      declarations that never paint in a static capture
                  (transition-*, animation-*, cursor, pointer-events, ...)
  decl_ablation   [optional] declarations overridden for every matched
                  element, found by in-page ablation
  merge_identical [optional] non-adjacent rules with identical declarations
                  merged into one selector list (cascade-checked in-page)
  link_remote     <link rel=stylesheet href=http(s)...> (never loads)

HOW A PAGE IS PROCESSED
  1. <style> blocks are located in the RAW html by regex, so the output is
     the input with only the CSS spans replaced: no BeautifulSoup round trip.
  2. The browser answers every question the classification needs
     (querySelectorAll per selector item, matchMedia per @media prelude,
     CSS.supports per @supports) on the loaded base page.
  3. Edits are planned as [start,end) deletions/replacements on the CSS
     text and applied at once; the in-page ORACLE (harness.ORACLE_JS: every
     element's rect + ink-relevant computed style + generated pseudo-
     elements) confirms nothing changed. If it did, the edit list is
     delta-debugged IN PAGE (no renders) and the offending edits are
     reported: they are classification bugs, not tolerated noise.
  4. The runner renders the result and requires pixel_identical + G1.

Equivalence is relative to the training render. Removing @media (max-width)
rules removes the page's other viewports; removing :hover removes its
interaction states. Level 2 already removed display:none subtrees. State the
invariant once, up front, in the paper.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
STATEFUL_PSEUDO = ("hover", "active", "focus-within", "focus-visible", "focus",
                   "visited", "target")
CONDITIONAL_AT = {"media", "supports", "layer", "container", "scope", "document",
                  "-moz-document"}
LEAF_AT_DEAD = {"keyframes": "keyframes", "-webkit-keyframes": "keyframes",
                "-moz-keyframes": "keyframes", "-o-keyframes": "keyframes",
                "page": "page"}
DEAD_PROPS = {"cursor", "pointer-events", "user-select", "-webkit-user-select",
              "-moz-user-select", "-ms-user-select", "touch-action",
              # caret-color: overridden by the harness FREEZE_CSS (!important),
              # so the page's value can never paint. will-change/resize are NOT
              # here: will-change:transform makes a containing block and a
              # stacking context, resize:none removes a textarea's grip.
              "-ms-touch-action", "scroll-behavior", "caret-color",
              "-webkit-tap-highlight-color", "-webkit-touch-callout",
              "-webkit-user-drag", "-webkit-overflow-scrolling",
              "print-color-adjust", "-webkit-print-color-adjust", "speak",
              "text-size-adjust", "-webkit-text-size-adjust", "-ms-text-size-adjust",
              "overscroll-behavior", "overscroll-behavior-x", "overscroll-behavior-y",
              "scroll-snap-type", "scroll-snap-align", "scroll-padding",
              "scroll-margin", "scroll-padding-top", "scroll-margin-top"}
DEAD_PROP_PREFIXES = ("transition", "animation", "-webkit-transition",
                      "-webkit-animation", "-moz-transition", "-moz-animation",
                      "-o-transition", "scroll-snap", "scroll-padding", "scroll-margin")

_PSEUDO_ELEMENT = re.compile(
    r"::?(before|after|first-line|first-letter|placeholder|selection|marker|"
    r"backdrop|file-selector-button|cue|spelling-error|grammar-error|"
    r"part\([^)]*\)|slotted\([^)]*\)|-webkit-[a-zA-Z-]+|-moz-[a-zA-Z-]+|"
    r"-ms-[a-zA-Z-]+)", re.I)

RE_STYLE_BLOCK = re.compile(r"(<style\b[^>]*>)(.*?)(</style\s*>)", re.S | re.I)
RE_LINK_TAG = re.compile(r"<link\b[^>]*>", re.I)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
@dataclass
class Decl:
    start: int
    end: int
    name: str
    value: str
    important: bool


@dataclass
class Rule:
    kind: str                 # style | at
    start: int
    end: int
    name: str = ""            # at-rule name (lowercase), "" for style rules
    prelude: str = ""         # source text of the prelude
    prelude_start: int = 0
    prelude_end: int = 0
    block_start: int = -1     # index of '{' or -1
    block_end: int = -1       # index of '}' or -1
    decls: list = field(default_factory=list)     # style rules only
    nested: bool = False      # style rule containing nested rules -> untouchable
    children: list = field(default_factory=list)  # conditional at-rules


@dataclass
class Edit:
    start: int
    end: int
    bucket: str
    replacement: str = ""
    detail: str = ""

    @property
    def chars(self) -> int:
        return (self.end - self.start) - len(self.replacement)


# ---------------------------------------------------------------------------
# Scanner: exact source offsets, strings/comments/url()/escapes aware
# ---------------------------------------------------------------------------
def normalize_newlines(css: str) -> str:
    """Same newline preprocessing the HTML parser applies to <style> text,
    so the raw block equals the DOM's textContent."""
    return css.replace("\r\n", "\n").replace("\r", "\n")


def _skip_string(s: str, i: int, hi: int) -> int:
    q = s[i]; i += 1
    while i < hi:
        c = s[i]
        if c == "\\":
            i += 2; continue
        if c == q:
            return i + 1
        if c == "\n":            # unterminated string ends at newline
            return i
        i += 1
    return hi


def _skip_comment(s: str, i: int, hi: int) -> int:
    j = s.find("*/", i + 2, hi)
    return hi if j < 0 else j + 2


def _skip_ws_comments(s: str, i: int, hi: int) -> int:
    while i < hi:
        if s[i] in " \t\n\f":
            i += 1
        elif s.startswith("/*", i):
            i = _skip_comment(s, i, hi)
        else:
            break
    return i


def _scan_to(s: str, i: int, stop: str, hi: int) -> int:
    """Index of the first char in `stop` at nesting depth 0 (parens,
    brackets) outside strings/comments/url(...); `hi` if none."""
    depth = 0
    while i < hi:
        c = s[i]
        if c in "\"'":
            i = _skip_string(s, i, hi); continue
        if s.startswith("/*", i):
            i = _skip_comment(s, i, hi); continue
        if c == "\\":
            i += 2; continue
        if depth == 0 and c in stop:
            return i
        if c in "([":
            if c == "(" and s[max(0, i - 3):i].lower() == "url":
                # unquoted url(...) may contain anything but ')'
                k = _skip_ws_comments(s, i + 1, hi)
                if k < hi and s[k] not in "\"'":
                    j = s.find(")", k, hi)
                    i = hi if j < 0 else j + 1
                    continue
            depth += 1
        elif c in ")]":
            depth = max(0, depth - 1)
        i += 1
    return hi


def _match_brace(s: str, i: int, hi: int) -> int:
    """i points at '{'. Returns index of the matching '}' or hi-1."""
    depth = 0
    while i < hi:
        c = s[i]
        if c in "\"'":
            i = _skip_string(s, i, hi); continue
        if s.startswith("/*", i):
            i = _skip_comment(s, i, hi); continue
        if c == "\\":
            i += 2; continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return hi - 1


_IDENT = re.compile(r"-?[A-Za-z_][\w-]*")


def parse_decls(s: str, lo: int, hi: int):
    """Declarations in a block body [lo,hi). Returns (decls, nested)."""
    decls, i = [], lo
    while True:
        i = _skip_ws_comments(s, i, hi)
        if i >= hi:
            break
        j = _scan_to(s, i, ";{}", hi)
        if j < hi and s[j] == "{":
            return decls, True            # nested rule inside a style rule
        text = s[i:j]
        end = j + 1 if (j < hi and s[j] == ";") else j
        colon = text.find(":")
        if colon > 0:
            name = text[:colon].strip().lower()
            value = text[colon + 1:].strip()
            imp = bool(re.search(r"!\s*important\s*$", value, re.I))
            if name:
                decls.append(Decl(i, end, name, value, imp))
        i = end
    return decls, False


def parse_rules(s: str, lo: int = 0, hi: int | None = None) -> list[Rule]:
    hi = len(s) if hi is None else hi
    rules, i = [], lo
    while True:
        i = _skip_ws_comments(s, i, hi)
        if i >= hi:
            break
        if s[i] == "@":
            m = _IDENT.match(s, i + 1)
            name = (m.group(0) if m else "").lower()
            j = _scan_to(s, i, "{;", hi)
            p0 = i + 1 + len(name)
            if j >= hi or s[j] == ";":
                rules.append(Rule("at", i, min(j + 1, hi), name, s[p0:j], p0, j))
                i = min(j + 1, hi)
                continue
            e = _match_brace(s, j, hi)
            r = Rule("at", i, e + 1, name, s[p0:j], p0, j, j, e)
            if name in CONDITIONAL_AT:
                r.children = parse_rules(s, j + 1, e)
            rules.append(r)
            i = e + 1
        else:
            j = _scan_to(s, i, "{;}", hi)
            if j >= hi or s[j] != "{":
                i = min(j + 1, hi)          # parse error: skip the junk
                continue
            e = _match_brace(s, j, hi)
            r = Rule("style", i, e + 1, "", s[i:j], i, j, j, e)
            r.decls, r.nested = parse_decls(s, j + 1, e)
            rules.append(r)
            i = e + 1
    return rules


# ---------------------------------------------------------------------------
# Selector lists
# ---------------------------------------------------------------------------
def split_selector_list(prelude: str) -> list[str]:
    """Top-level comma split (commas inside (), [] and strings are kept)."""
    items, depth, start, i, n = [], 0, 0, 0, len(prelude)
    while i < n:
        c = prelude[i]
        if c in "\"'":
            i = _skip_string(prelude, i, n); continue
        if c == "\\":
            i += 2; continue
        if c in "([":
            depth += 1
        elif c in ")]":
            depth = max(0, depth - 1)
        elif c == "," and depth == 0:
            items.append(prelude[start:i]); start = i + 1
        i += 1
    items.append(prelude[start:])
    return [x.strip() for x in items if x.strip()]


def is_stateful(item: str) -> bool:
    """True if a :hover-like pseudo-class occurs at depth 0 (a :not(:hover)
    is TRUE in the static state and is left to the match test)."""
    depth, i, n = 0, 0, len(item)
    while i < n:
        c = item[i]
        if c in "\"'":
            i = _skip_string(item, i, n); continue
        if c == "\\":
            i += 2; continue
        if c in "([":
            depth += 1
        elif c in ")]":
            depth = max(0, depth - 1)
        elif c == ":" and depth == 0 and not item.startswith("::", i):
            m = _IDENT.match(item, i + 1)
            if m and m.group(0).lower() in STATEFUL_PSEUDO \
                    and not item.startswith("(", i + 1 + len(m.group(0))):
                return True
        i += 1
    return False


def strip_comments(text: str) -> str:
    out, i, n = [], 0, len(text)
    while i < n:
        if text[i] in "\"'":
            j = _skip_string(text, i, n); out.append(text[i:j]); i = j
        elif text.startswith("/*", i):
            i = _skip_comment(text, i, n)
        else:
            out.append(text[i]); i += 1
    return "".join(out)


_PSEUDO_ELEMENT_NAMES = {"before", "after", "first-line", "first-letter", "placeholder",
                         "selection", "marker", "backdrop", "file-selector-button", "cue",
                         "spelling-error", "grammar-error", "part", "slotted",
                         "target-text", "highlight", "view-transition"}


def matchable(item: str) -> str:
    """Selector with pseudo-elements removed, for querySelectorAll. Walks
    the selector so that quoted attribute values ([data-x="::before"]) and
    the insides of functional pseudo-classes are never rewritten."""
    out, i, n = [], 0, len(item)
    while i < n:
        c = item[i]
        if c in "\"'":
            j = _skip_string(item, i, n); out.append(item[i:j]); i = j; continue
        if c == "\\":
            out.append(item[i:i + 2]); i += 2; continue
        if c == "[":
            j = _scan_to(item, i + 1, "]", n); out.append(item[i:min(j + 1, n)]); i = j + 1; continue
        if c == ":":
            dbl = item.startswith("::", i)
            m = _IDENT.match(item, i + (2 if dbl else 1))
            name = m.group(0).lower() if m else ""
            is_pe = dbl or name in {"before", "after", "first-line", "first-letter"}
            if name and is_pe and (dbl and name.startswith("-") or name in _PSEUDO_ELEMENT_NAMES
                                   or name.startswith(("-webkit-", "-moz-", "-ms-"))):
                i = m.end()
                if i < n and item[i] == "(":            # ::part(x), ::slotted(y)
                    i = _scan_to(item, i + 1, ")", n) + 1
                continue
        out.append(c); i += 1
    return "".join(out).strip()


def is_dead_prop(name: str) -> bool:
    return name in DEAD_PROPS or name.startswith(DEAD_PROP_PREFIXES)


# ---------------------------------------------------------------------------
# Browser questions
# ---------------------------------------------------------------------------
QUESTIONS_JS = """
(q) => {
  const sel = q.selectors.map(s => { try { return document.querySelectorAll(s).length; }
                                     catch (e) { return -1; } });
  const media = q.media.map(m => { try { return matchMedia(m).matches; } catch (e) { return true; } });
  const supports = q.supports.map(c => { try { return CSS.supports(c); } catch (e) { return true; } });
  return { selectors: sel, media: media, supports: supports };
}
"""

PAGE_STYLES = "style:not([data-harness-freeze])"
GET_STYLES_JS = f"() => Array.from(document.querySelectorAll('{PAGE_STYLES}')).map(s => s.textContent)"
SET_STYLES_JS = f"""
(texts) => {{ const els = document.querySelectorAll('{PAGE_STYLES}');
             if (els.length !== texts.length) return false;
             for (let i = 0; i < els.length; i++) if (els[i].textContent !== texts[i]) els[i].textContent = texts[i];
             return true; }}
"""


def collect_questions(rules: list[Rule], q: dict) -> None:
    for r in rules:
        if r.kind == "style":
            for it in split_selector_list(strip_comments(r.prelude)):
                ms = matchable(it)
                if ms:
                    q["selectors"].setdefault(ms, None)
        elif r.name == "media":
            q["media"].setdefault(r.prelude.strip(), None)
        elif r.name == "supports":
            q["supports"].setdefault(r.prelude.strip(), None)
        if r.children:
            collect_questions(r.children, q)


def ask_browser(harness, q: dict) -> dict:
    keys = {k: list(v.keys()) for k, v in q.items()}
    ans = harness.page.evaluate(QUESTIONS_JS, keys)
    return {k: dict(zip(keys[k], ans[k])) for k in keys}


# ---------------------------------------------------------------------------
# Planning
# ---------------------------------------------------------------------------
_URL_RE = re.compile(r"""url\(\s*(?:"([^"]*)"|'([^']*)'|([^)\s]*))\s*\)|(?<![\w-])"([^"]*)"|'([^']*)'""", re.I)


def _urls_in(text: str) -> list[str]:
    return [next(g for g in m.groups() if g is not None) for m in _URL_RE.finditer(text)]


def url_reachable(url: str, base_dir) -> bool:
    """Can this URL load under the harness? http(s) is aborted; data: is
    inline; anything else is resolved as a file next to the page (the
    harness lets file:// through) and counts as reachable iff it exists.
    Unknown/unresolvable -> reachable (conservative: keep the rule)."""
    u = url.strip().lower()
    if not u:
        return False
    if u.startswith(("http://", "https://", "//")):
        return False
    if u.startswith("data:"):
        return True
    if base_dir is None:
        return True
    from pathlib import Path
    from urllib.parse import unquote, urlsplit
    if u.startswith("file://"):
        path = Path(unquote(urlsplit(url).path))
    else:
        path = Path(base_dir) / unquote(urlsplit(url).path)
    try:
        return path.is_file()
    except OSError:
        return True


def _font_face_dead(r: Rule, s: str, base_dir) -> bool:
    body = s[r.block_start + 1:r.block_end]
    if "local(" in body.lower():
        return False
    urls = _urls_in(strip_comments(body))
    return bool(urls) and not any(url_reachable(u, base_dir) for u in urls)


def _import_dead(r: Rule, base_dir) -> bool:
    urls = _urls_in(strip_comments(r.prelude))
    return bool(urls) and not any(url_reachable(u, base_dir) for u in urls)


def plan_rules(rules: list[Rule], s: str, ans: dict, opts: dict) -> tuple[list[Edit], bool]:
    """Returns (edits, all_dead) for a rule list. Whole-block deletions are
    emitted instead of their children's edits."""
    edits, all_dead = [], True
    for r in rules:
        dead_bucket = None
        if r.kind == "style":
            items = split_selector_list(strip_comments(r.prelude))
            live, dead_items, dead_kinds = [], [], set()
            for it in items:
                ms = matchable(it)
                n = ans["selectors"].get(ms, -1) if ms else -1
                if n == 0:
                    dead_items.append(it)
                    dead_kinds.add("stateful" if is_stateful(it) else "unmatched")
                else:
                    live.append(it)
            if items and not live:
                dead_bucket = "stateful" if dead_kinds == {"stateful"} else "unmatched"
            elif dead_items and opts.get("trim_selectors", True):
                edits.append(Edit(r.prelude_start, r.prelude_end, "selector_trim",
                                  ",".join(live), f"dropped {len(dead_items)} of {len(items)}"))
            if dead_bucket is None:
                all_dead = False
                if not r.nested and opts.get("dead_props", True):
                    for d in r.decls:
                        if is_dead_prop(d.name):
                            edits.append(Edit(d.start, d.end, "dead_props", "", d.name))
        else:
            if r.name == "media":
                if ans["media"].get(r.prelude.strip(), True) is False:
                    dead_bucket = "media_nomatch"
            elif r.name == "supports":
                if ans["supports"].get(r.prelude.strip(), True) is False:
                    dead_bucket = "supports_nomatch"
            elif r.name in LEAF_AT_DEAD:
                dead_bucket = LEAF_AT_DEAD[r.name]
            elif r.name == "font-face":
                if _font_face_dead(r, s, opts.get("base_dir")):
                    dead_bucket = "font_face_remote"
            elif r.name == "import":
                if _import_dead(r, opts.get("base_dir")):
                    dead_bucket = "import"
            if dead_bucket is None and r.name in CONDITIONAL_AT and r.block_start >= 0:
                sub, sub_all_dead = plan_rules(r.children, s, ans, opts)
                layer_name = r.prelude.strip() if r.name == "layer" else ""
                if sub_all_dead and r.children and layer_name:
                    # The first appearance of a named layer fixes its place in
                    # the layer order; an emptied block must keep that place.
                    edits.append(Edit(r.start, r.end, "block_emptied",
                                      "@layer " + layer_name + ";", "@layer " + layer_name))
                    all_dead = False
                elif sub_all_dead and r.children:
                    dead_bucket = "block_emptied"
                else:
                    edits.extend(sub)
                    all_dead = False
            elif dead_bucket is None:
                all_dead = False            # @charset/@namespace/@property/...
        if dead_bucket is not None:
            edits.append(Edit(r.start, r.end, dead_bucket, "",
                              (r.prelude if r.kind == "style" else "@" + r.name + r.prelude)[:80]))
    return edits, all_dead


def apply_edits(s: str, edits: list[Edit]) -> str:
    out = s
    for e in sorted(edits, key=lambda e: e.start, reverse=True):
        out = out[:e.start] + e.replacement + out[e.end:]
    return out


# ---------------------------------------------------------------------------
# Style blocks in the raw HTML
# ---------------------------------------------------------------------------
@dataclass
class StyleBlock:
    start: int      # offset of the CSS text in the html
    end: int
    css: str        # newline-normalized CSS


def find_style_blocks(html: str) -> list[StyleBlock]:
    return [StyleBlock(m.start(2), m.end(2), normalize_newlines(m.group(2)))
            for m in RE_STYLE_BLOCK.finditer(html)]


def splice_blocks(html: str, blocks: list[StyleBlock], new_css: list[str]) -> str:
    out = html
    for b, css in sorted(zip(blocks, new_css), key=lambda t: t[0].start, reverse=True):
        out = out[:b.start] + css + out[b.end:]
    return out


def remote_link_edits(html: str) -> list[tuple[int, int]]:
    """Spans of <link rel=stylesheet href=http(s)...> tags."""
    spans = []
    for m in RE_LINK_TAG.finditer(html):
        tag = m.group(0).lower()
        if "stylesheet" in tag and re.search(r"href\s*=\s*[\"']?https?://", tag):
            spans.append((m.start(), m.end()))
    return spans


# ---------------------------------------------------------------------------
# In-page checks
# ---------------------------------------------------------------------------
def oracle_same(harness, texts: list[str]) -> tuple[bool, dict]:
    if not harness.page.evaluate(SET_STYLES_JS, texts):
        return False, {"first": "style count mismatch"}
    res = harness.oracle_compare()
    return (res["nDiff"] == 0 and not res["missing"] and not res["added"]), res


class InPageResolver:
    """Delta-debugging over CSS edits using the oracle only (no renders)."""

    def __init__(self, harness, blocks, rules_css, budget=200):
        self.h, self.blocks, self.base = harness, blocks, rules_css
        self.budget, self.calls = budget, 0
        self.blacklist: list[tuple[Edit, str]] = []

    def texts_for(self, per_block_edits: list[list[Edit]]) -> list[str]:
        return [apply_edits(css, eds) for css, eds in zip(self.base, per_block_edits)]

    def test(self, flat: list[tuple[int, Edit]]) -> bool | None:
        if self.calls >= self.budget:
            return None
        self.calls += 1
        per = [[] for _ in self.blocks]
        for bi, e in flat:
            per[bi].append(e)
        ok, self.last = oracle_same(self.h, self.texts_for(per))
        return ok

    def resolve(self, flat):
        if not flat:
            return []
        ok = self.test(flat)
        if ok:
            return flat
        if ok is None:
            return []
        if len(flat) == 1:
            self.blacklist.append((flat[0][1], self.last.get("first", "")))
            return []
        mid = len(flat) // 2
        left = self.resolve(flat[:mid])
        right = self.resolve(flat[mid:])
        if left and right:
            ok2 = self.test(left + right)
            if ok2:
                return left + right
            return left if len(left) >= len(right) else right
        return left + right


# ---------------------------------------------------------------------------
# Optional operators (both oracle-checked per candidate)
# ---------------------------------------------------------------------------
def ablate_declarations(harness, texts: list[str], max_trials: int) -> tuple[list[str], list[dict]]:
    """Greedy: delete one declaration of a live style rule at a time and
    keep the deletion iff the oracle sees no change. Sequential, so each
    accepted deletion is tested on top of the previous ones."""
    texts, log, trials = list(texts), [], 0
    for bi in range(len(texts)):
        while True:
            # Re-parse after every accepted deletion (offsets move).
            cands = []

            def walk(rules):
                for r in rules:
                    if r.kind == "style" and not r.nested:
                        cands.extend((d, r) for d in r.decls if not is_dead_prop(d.name))
                    walk(r.children)
            walk(parse_rules(texts[bi]))
            progressed = False
            for d, r in cands:
                if trials >= max_trials:
                    return texts, log
                if any(l["block"] == bi and l["start"] == d.start and l["name"] == d.name
                       and not l["kept"] for l in log):
                    continue
                trials += 1
                trial = list(texts)
                trial[bi] = apply_edits(texts[bi], [Edit(d.start, d.end, "decl_ablation")])
                ok, res = oracle_same(harness, trial)
                log.append({"block": bi, "start": d.start, "name": d.name,
                            "selector": r.prelude.strip()[:60], "kept": ok,
                            "chars": d.end - d.start, "first": "" if ok else res.get("first", "")})
                if ok:
                    texts = trial; progressed = True
                    break                  # offsets changed: re-parse
            if not progressed:
                break
    return texts, log


def merge_identical_rules(harness, texts: list[str]) -> tuple[list[str], list[dict]]:
    """Merge NON-adjacent style rules (same parent block) whose declaration
    text is identical into the LAST member's position. Each group is
    oracle-checked: moving a rule later changes the cascade whenever an
    intervening equal-specificity rule co-matches -- the oracle sees that as
    a computed-style diff and the group is rejected."""
    texts, log = list(texts), []
    for bi in range(len(texts)):
        while True:
            groups: dict = {}

            def walk(rules, parent_key):
                for idx, r in enumerate(rules):
                    if r.kind == "style" and not r.nested and r.decls:
                        # Exact body text (ends trimmed): collapsing internal
                        # whitespace would equate content:"A B" with "AB".
                        key = (parent_key, texts[bi][r.block_start + 1:r.block_end].strip())
                        groups.setdefault(key, []).append(r)
                    if r.children:
                        walk(r.children, parent_key + (idx,))
            walk(parse_rules(texts[bi]), ())
            done = False
            for key, members in groups.items():
                if len(members) < 2:
                    continue
                if any(l["block"] == bi and l["key"] == key[1][:40] and not l["kept"] for l in log):
                    continue
                last = members[-1]
                sels = [m.prelude.strip() for m in members]
                eds = [Edit(m.start, m.end, "merge_identical") for m in members[:-1]]
                eds.append(Edit(last.prelude_start, last.prelude_end, "merge_identical",
                                ",".join(sels)))
                trial = list(texts)
                trial[bi] = apply_edits(texts[bi], eds)
                ok, res = oracle_same(harness, trial)
                log.append({"block": bi, "key": key[1][:40], "n": len(members), "kept": ok,
                            "chars": len(texts[bi]) - len(trial[bi]),
                            "first": "" if ok else res.get("first", "")})
                if ok:
                    texts = trial; done = True
                    break
            if not done:
                break
    return texts, log


# ---------------------------------------------------------------------------
# Page-level driver (no rendering here; the runner renders and gates)
# ---------------------------------------------------------------------------
def level3_page(harness, html: str, opts: dict) -> dict:
    """The page must already be LOADED in the harness (harness.load(base)).
    Returns {status, html, edits, blacklist, buckets_chars, ...}."""
    blocks = find_style_blocks(html)
    out = {"status": "ok", "n_blocks": len(blocks)}
    if not blocks:
        new_html, spans = html, []
        if opts.get("drop_remote_links", True):
            spans = remote_link_edits(html)
            for a, b in sorted(spans, reverse=True):
                new_html = new_html[:a] + new_html[b:]
        return {**out, "status": "ok" if spans else "no_css", "html": new_html,
                "n_edits_planned": len(spans), "n_edits_kept": len(spans), "oracle_calls": 0,
                "blacklist": [], "css_chars_before": 0, "css_chars_after": 0,
                "buckets_chars": {"link_remote": sum(b - a for a, b in spans)} if spans else {}}

    dom_texts = harness.page.evaluate(GET_STYLES_JS)
    if len(dom_texts) != len(blocks) or any(d != b.css for d, b in zip(dom_texts, blocks)):
        return {**out, "status": "style_mismatch", "html": html}

    harness.oracle_baseline()
    base_css = [b.css for b in blocks]
    all_rules = [parse_rules(css) for css in base_css]

    q = {"selectors": {}, "media": {}, "supports": {}}
    for rules in all_rules:
        collect_questions(rules, q)
    ans = ask_browser(harness, q)

    per_block = [plan_rules(rules, css, ans, opts)[0]
                 for rules, css in zip(all_rules, base_css)]
    flat = [(bi, e) for bi, eds in enumerate(per_block) for e in eds]
    out["n_edits_planned"] = len(flat)

    res = InPageResolver(harness, blocks, base_css, budget=opts.get("oracle_budget", 200))
    kept = res.resolve(flat) if flat else []
    out.update({"n_edits_kept": len(kept), "oracle_calls": res.calls,
                "blacklist": [{"bucket": e.bucket, "detail": e.detail, "first": why}
                              for e, why in res.blacklist]})
    per_kept = [[] for _ in blocks]
    for bi, e in kept:
        per_kept[bi].append(e)
    texts = res.texts_for(per_kept)

    buckets: dict[str, int] = {}
    for _, e in kept:
        buckets[e.bucket] = buckets.get(e.bucket, 0) + e.chars

    if opts.get("ablate_declarations"):
        texts, log = ablate_declarations(harness, texts, opts.get("max_ablation", 600))
        c = sum(l["chars"] for l in log if l["kept"])
        buckets["decl_ablation"] = buckets.get("decl_ablation", 0) + c
        out["ablation_log"] = log
    if opts.get("merge_identical"):
        texts, log = merge_identical_rules(harness, texts)
        c = sum(l["chars"] for l in log if l["kept"])
        buckets["merge_identical"] = buckets.get("merge_identical", 0) + c
        out["merge_log"] = log

    # Leave the page as we found it (the runner reloads anyway).
    harness.page.evaluate(SET_STYLES_JS, base_css)

    new_html = splice_blocks(html, blocks, texts)
    if opts.get("drop_remote_links", True):
        spans = remote_link_edits(new_html)
        for a, b in sorted(spans, reverse=True):
            new_html = new_html[:a] + new_html[b:]
        if spans:
            buckets["link_remote"] = sum(b - a for a, b in spans)
    out.update({"html": new_html, "buckets_chars": buckets,
                "css_chars_before": sum(len(c) for c in base_css),
                "css_chars_after": sum(len(t) for t in texts)})
    return out


def resolve_base(ROOT, src, pid, r, prefer: str, final_df, sel_df):
    """(path, label). prefer='final': step 08's entry for this page decides,
    INCLUDING an 'original' entry (08 may have rejected every compressed
    rung; the training target is then the original and L3 must start
    there). A page with no step-08 entry falls back to step 07's selection
    with the reason in the label; 'selected' skips 08; 'original' uses the
    raw file. The frozen target's recorded hash is verified when present."""
    import hashlib
    from pathlib import Path
    if prefer == "original":
        return Path(r["html_path"]), "original"
    if prefer == "final":
        if final_df is not None and pid in final_df.index:
            row = final_df.loc[pid]
            lvl = str(row.get("level", "original"))
            p = Path(str(row.get("html_path", "")))
            if lvl == "original":
                return Path(r["html_path"]), "final:original"
            if p.exists():
                want = str(row.get("final_sha256", "") or "")
                if want and want != "nan":
                    got = hashlib.sha256(p.read_bytes()).hexdigest()
                    if got != want:
                        raise ValueError(f"step-08 target {p.name} hash mismatch "
                                         f"(file changed since validation)")
                return p, "final:" + lvl
            raise FileNotFoundError(f"step-08 target missing: {p}")
        note = " (no step-08 entry)" if final_df is not None else " (no step-08 run)"
    else:
        note = ""
    if sel_df is not None and pid in sel_df.index:
        row = sel_df.loc[pid]
        p = Path(str(row.get("html_path", "")))
        if str(row.get("level", "original")) != "original" and p.exists():
            return p, "selected:" + str(row["level"]) + note
    return Path(r["html_path"]), "original" + note


def load_base_tables(ROOT, src):
    import pandas as pd
    out = []
    for f in (ROOT / "data" / "splits" / f"compressed_{src}_manifest.csv",
              ROOT / "reports" / "csv" / f"level_selection_{src}.csv"):
        if f.exists():
            d = pd.read_csv(f); d["page_id"] = d["page_id"].astype(str)
            out.append(d.set_index("page_id"))
        else:
            out.append(None)
    return out[0], out[1]