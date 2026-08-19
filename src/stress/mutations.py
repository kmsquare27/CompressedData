"""Mutation library for the gate stress test (protocol Stage 2.3).

Purpose: prove the efficacy of gates G0-G6. A sound gate must REJECT every
known-breaking edit and ACCEPT every known-safe edit; per-metric detection
on this labeled benchmark is what calibrates thresholds and what shows,
quantitatively, that SSIM alone is insufficient.

Design rules (all three were violated by the v1 library -- see STRESS_TEST.md):

  1. ANNOTATION-DRIVEN TARGETING. Every operator receives PageMeta (browser
     facts collected on the rendered stamped page) and selects targets that
     are verified-paintable, editing RELATIVE to computed values. A soup-only
     operator can label an invisible no-op edit "breaking" and poison recall.
  2. SEVERITY IS A VARIABLE, NOT A CONSTANT. M3 shifts {8,16,32}px, M4
     recolors to CIEDE2000 ~ {2,5,10}, M5 resizes fonts {-2,+2}px. Near-
     threshold probes are exactly where thresholds get calibrated and where
     SSIM's blindness is demonstrated.
  3. DETERMINISM. Operators draw all randomness from the rng handed in by
     the runner (seeded with a stable CRC, not Python's salted hash()) and
     iterate candidates in document order, so a re-run reproduces every
     mutant byte-for-byte.

Every operator: fn(html, meta, rng, params, exclude_ids) -> MutationOutcome
or None (None = not applicable on this page; the runner logs it as NA).
`html` is the STAMPED original; ids in `exclude_ids` were duds on earlier
attempts and must not be re-targeted.

Registry intents:
  breaking  -> must change pixels; verified by render-back (dud otherwise)
  safe      -> must NOT change pixels; verified (finding otherwise)
  probe     -> historically-assumed-safe edits under test (excluded from
               calibration; their verified break-rate is itself a result).
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
from bs4 import BeautifulSoup, Comment, NavigableString

from src.stress.annotate import STAMP_ATTR

ATOMIC = {"IMG", "INPUT", "BUTTON", "TEXTAREA", "SELECT", "SVG", "VIDEO", "HR"}
_NEVER_TOUCH = {"HTML", "BODY", "SCRIPT", "STYLE", "HEAD", "TITLE", "META", "LINK"}


@dataclass
class MutationOutcome:
    html: str
    target_ids: tuple = ()
    severity_nominal: float | None = None
    severity_achieved: float | None = None
    detail: str = ""


@dataclass(frozen=True)
class MutationSpec:
    name: str
    intent: str                      # breaking | safe | probe
    family: str                      # structure | text | geometry | color | typography | source
    fn: object
    variants: tuple = (("default", ()),)   # ((key, ((param, value), ...)), ...)
    protocol_id: str = ""
    extension: bool = False          # True = beyond the protocol's M1-M7/S1-S6 list


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _soup(html: str) -> BeautifulSoup:
    return BeautifulSoup(html, "lxml")


def _el(soup, mid: int):
    return soup.find(attrs={STAMP_ATTR: str(mid)})


def _add_style(tag, css: str) -> None:
    prev = (tag.get("style") or "").strip().rstrip(";")
    tag["style"] = (prev + "; " if prev else "") + css


def _pick(rng, cands: list):
    """Deterministic choice: cands must already be in document order."""
    return cands[rng.randrange(len(cands))]


def _area(e) -> float:
    return float(e["w"]) * float(e["h"])


def _preview(e) -> str:
    return f"<{e['tag'].lower()} id={e['id']} {int(e['w'])}x{int(e['h'])}>"


# ---------------------------------------------------------------------------
# Color machinery for M4: find an sRGB color at a TARGET CIEDE2000 distance
# from the element's real computed background. A fixed magenta (v1) has no
# severity axis and can no-op if the element is invisible; this searches the
# Lab space along several directions with binary search, clamps to gamut,
# and reports the ACHIEVED distance of the final quantized sRGB color.
# ---------------------------------------------------------------------------
_CSS_RGB = re.compile(r"rgba?\(([^)]+)\)")


def parse_css_color(s: str):
    """'rgb(1,2,3)' / 'rgba(1,2,3,0.5)' / '#aabbcc' -> (r,g,b,a) or None."""
    s = (s or "").strip().lower()
    if not s or s == "transparent":
        return None
    if s.startswith("#"):
        h = s[1:]
        if len(h) == 3:
            h = "".join(c * 2 for c in h)
        if len(h) >= 6:
            try:
                return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16), 1.0)
            except ValueError:
                return None
        return None
    m = _CSS_RGB.match(s)
    if not m:
        return None
    parts = [p.strip() for p in m.group(1).replace("/", ",").split(",") if p.strip()]
    try:
        r, g, b = (float(parts[0]), float(parts[1]), float(parts[2]))
        a = float(parts[3]) if len(parts) > 3 else 1.0
    except (ValueError, IndexError):
        return None
    return (r, g, b, a)


def _rgb_to_lab(rgb) -> np.ndarray:
    from skimage import color as skcolor
    arr = np.asarray(rgb, dtype=float).reshape(1, 1, 3) / 255.0
    return skcolor.rgb2lab(arr)[0, 0]


def _lab_to_rgb255(lab: np.ndarray):
    import warnings
    from skimage import color as skcolor
    with warnings.catch_warnings():
        # out-of-gamut probes are expected: we clamp and re-measure achieved
        warnings.simplefilter("ignore")
        rgb = skcolor.lab2rgb(np.asarray(lab, dtype=float).reshape(1, 1, 3))[0, 0]
    return tuple(int(v) for v in np.clip(np.round(rgb * 255), 0, 255))


def _de00(lab1: np.ndarray, lab2: np.ndarray) -> float:
    from skimage import color as skcolor
    return float(skcolor.deltaE_ciede2000(
        np.asarray(lab1, float).reshape(1, 1, 3),
        np.asarray(lab2, float).reshape(1, 1, 3))[0, 0])


# Lab-space directions: +b, -b, +a, -a, -L, +L (chromatic moves first so
# near-neutral targets change hue, not lightness, when possible).
_LAB_DIRS = [np.array(d, float) for d in
             [(0, 0, 1), (0, 0, -1), (0, 1, 0), (0, -1, 0),
              (-1, 0, 0), (1, 0, 0)]]


def color_at_delta_e(rgb, target: float, tol: float = 0.5):
    """Return (new_rgb, achieved_de00) with achieved ~= target from `rgb`,
    or None if no direction can reach the target inside the sRGB gamut.
    `achieved` is measured on the final quantized sRGB color -- what the
    browser will actually paint."""
    base = _rgb_to_lab(rgb[:3])

    def probe(direction, t):
        cand_rgb = _lab_to_rgb255(base + direction * t)
        return cand_rgb, _de00(base, _rgb_to_lab(cand_rgb))

    best = None
    for d in _LAB_DIRS:
        lo, hi = 0.0, 60.0
        _, a_hi = probe(d, hi)
        if a_hi < target - tol:          # gamut clamp: direction can't reach
            continue
        got = None
        for _ in range(40):
            mid = (lo + hi) / 2.0
            cand, a_mid = probe(d, mid)
            if abs(a_mid - target) <= tol:
                got = (cand, round(a_mid, 3))
                break
            if a_mid < target:
                lo = mid
            else:
                hi = mid
        if got is None:
            cand, a_mid = probe(d, (lo + hi) / 2.0)
            got = (cand, round(a_mid, 3))
        if best is None or abs(got[1] - target) < abs(best[1] - target):
            best = got
        if abs(best[1] - target) <= tol:
            return best
    if best is not None and abs(best[1] - target) <= max(tol, 0.25 * target):
        return best
    return None


# ===========================================================================
# BREAKING operators (M1-M7 from the protocol, +M8 extension)
# ===========================================================================
def m1_delete_text_block(html, meta, rng, params, exclude):
    """M1: delete one verified-visible text-bearing block."""
    cands = meta.candidates(
        lambda e: e["paintable"] and e["leafTextLen"] >= 3
        and e["tag"] not in _NEVER_TOUCH and _area(e) >= 16, exclude)
    if not cands:
        return None
    t = _pick(rng, cands)
    s = _soup(html)
    tag = _el(s, t["id"])
    if tag is None:
        return None
    tag.decompose()
    return MutationOutcome(str(s), (t["id"],), detail=f"deleted {_preview(t)}")


_LETTERS = "abcdefghijklmnopqrstuvwxyz"


def _twist_char(ch: str) -> str:
    """A different character of the same class (letter->next letter same
    case, digit->next digit) so length and character class are preserved:
    this is a pure TEXT change, the failure mode G3 exists to catch."""
    if ch.islower():
        return _LETTERS[(_LETTERS.index(ch) + 1) % 26] if ch in _LETTERS else "x"
    if ch.isupper():
        low = ch.lower()
        return (_LETTERS[(_LETTERS.index(low) + 1) % 26].upper()
                if low in _LETTERS else "X")
    if ch.isdigit():
        return str((int(ch) + 1) % 10)
    return ch


def m2_corrupt_text(html, meta, rng, params, exclude):
    """M2: replace ~30% of alphanumeric characters in one element's DIRECT
    text nodes. v1 used tag.string = ..., which silently DELETED all child
    elements (a structural change masquerading as a text change); mutating
    only NavigableString children keeps the DOM shape intact."""
    cands = meta.candidates(
        lambda e: e["paintable"] and e["leafTextLen"] >= 10
        and e["tag"] not in _NEVER_TOUCH, exclude)
    if not cands:
        return None
    t = _pick(rng, cands)
    s = _soup(html)
    tag = _el(s, t["id"])
    if tag is None:
        return None
    strs = [c for c in tag.children
            if isinstance(c, NavigableString) and not isinstance(c, Comment)
            and c.strip()]
    flat = [(si, ci) for si, ns in enumerate(strs)
            for ci, ch in enumerate(str(ns)) if ch.isalnum()]
    if len(flat) < 8:
        return None
    k = max(1, int(round(0.30 * len(flat))))
    chosen = set(map(tuple, rng.sample(flat, k)))
    for si, ns in enumerate(strs):
        chars = list(str(ns))
        for ci in range(len(chars)):
            if (si, ci) in chosen:
                chars[ci] = _twist_char(chars[ci])
        ns.replace_with(NavigableString("".join(chars)))
    return MutationOutcome(str(s), (t["id"],), severity_nominal=0.30,
                           detail=f"corrupted {k}/{len(flat)} chars in {_preview(t)}")


def m3_shift_element(html, meta, rng, params, exclude):
    """M3: add margin-left {8,16,32}px to a verified-visible in-flow element
    (relative to its COMPUTED margin, !important so stylesheets can't undo
    it)."""
    px = dict(params)["px"]
    cands = meta.candidates(
        lambda e: e["paintable"] and e["position"] in ("static", "relative")
        and e["tag"] not in _NEVER_TOUCH and _area(e) >= 16, exclude)
    if not cands:
        return None
    t = _pick(rng, cands)
    s = _soup(html)
    tag = _el(s, t["id"])
    if tag is None:
        return None
    new_ml = t["mar"][3] + px
    _add_style(tag, f"margin-left:{new_ml:g}px !important")
    return MutationOutcome(str(s), (t["id"],), severity_nominal=float(px),
                           detail=f"margin-left {t['mar'][3]:g}->{new_ml:g}px on {_preview(t)}")


def m4_recolor_element(html, meta, rng, params, exclude):
    """M4: shift one element's background color to CIEDE2000 ~= {2,5,10}
    from its real computed background. Severity-graded color probes are what
    calibrate G5 and expose SSIM's equal-luminance blindness."""
    target = dict(params)["target_de"]
    cands = meta.candidates(
        lambda e: e["paintable"] and _area(e) >= 100
        and e["tag"] not in _NEVER_TOUCH
        and (parse_css_color(e["bg"]) or (0, 0, 0, 0))[3] >= 0.999, exclude)
    if not cands:
        return None
    order = list(cands)
    rng.shuffle(order)
    for t in order[:10]:
        rgb = parse_css_color(t["bg"])
        found = color_at_delta_e(rgb, float(target))
        if found is None:
            continue
        (r, g, b), achieved = found
        s = _soup(html)
        tag = _el(s, t["id"])
        if tag is None:
            continue
        _add_style(tag, f"background-color:rgb({r},{g},{b}) !important")
        return MutationOutcome(
            str(s), (t["id"],), severity_nominal=float(target),
            severity_achieved=achieved,
            detail=f"bg {t['bg']} -> rgb({r},{g},{b}) dE00={achieved} on {_preview(t)}")
    return None


def m5_fontsize_bump(html, meta, rng, params, exclude):
    """M5: font-size {-2,+2}px RELATIVE to the computed size (v1 forced an
    absolute 23px, a literal no-op on any element already at 23px)."""
    delta = dict(params)["delta"]
    cands = meta.candidates(
        lambda e: e["paintable"] and e["leafTextLen"] >= 3
        and e["tag"] not in _NEVER_TOUCH
        and e["fontSize"] + delta >= 7, exclude)
    if not cands:
        return None
    t = _pick(rng, cands)
    s = _soup(html)
    tag = _el(s, t["id"])
    if tag is None:
        return None
    new_fs = t["fontSize"] + delta
    _add_style(tag, f"font-size:{new_fs:g}px !important")
    return MutationOutcome(str(s), (t["id"],), severity_nominal=float(delta),
                           detail=f"font-size {t['fontSize']:g}->{new_fs:g}px on {_preview(t)}")


def m6_remove_loadbearing_wrapper(html, meta, rng, params, exclude):
    """M6: unwrap one wrapper that provably contributes layout/paint
    (computed padding, margin, border or background). v1 only matched
    inline style="padding..." and therefore found ~nothing on real-world
    (stylesheet-styled) pages -- the exact stratum this failure mode
    matters for."""
    def loadbearing(e):
        bg = parse_css_color(e["bg"])
        return (e["paintable"] and e["nChildElems"] >= 1
                and e["tag"] not in _NEVER_TOUCH and e["tag"] not in ATOMIC
                and (sum(e["pad"]) > 0 or sum(e["mar"]) > 0
                     or sum(e["borderW"]) > 0
                     or (bg is not None and bg[3] > 0.01)))
    cands = meta.candidates(loadbearing, exclude)
    if not cands:
        return None
    t = _pick(rng, cands)
    s = _soup(html)
    tag = _el(s, t["id"])
    if tag is None:
        return None
    tag.unwrap()
    return MutationOutcome(str(s), (t["id"],),
                           detail=f"unwrapped load-bearing {_preview(t)}")


def m7_swap_siblings(html, meta, rng, params, exclude):
    """M7: swap two substantial visible sibling blocks (scene-level
    rearrangement -- same content, wrong order; G4's center-shift/IoU must
    catch it even though G3's text metrics may not on symmetric text)."""
    by_parent: dict = {}
    for e in meta.candidates(
            lambda e: e["paintable"] and _area(e) >= 100
            and e["tag"] not in _NEVER_TOUCH and e["parentId"] >= 0, exclude):
        by_parent.setdefault(e["parentId"], []).append(e)
    parents = [pid for pid in sorted(by_parent) if len(by_parent[pid]) >= 2]
    if not parents:
        return None
    pid = _pick(rng, parents)
    kids = by_parent[pid]
    i, j = rng.sample(range(len(kids)), 2)
    a_id, b_id = kids[i]["id"], kids[j]["id"]
    s = _soup(html)
    a, b = _el(s, a_id), _el(s, b_id)
    if a is None or b is None or str(a) == str(b):
        return None
    ph = s.new_tag("template")
    a.replace_with(ph)
    b.replace_with(a)
    ph.replace_with(b)
    return MutationOutcome(str(s), (a_id, b_id),
                           detail=f"swapped siblings id={a_id}<->id={b_id} under id={pid}")


def m8_delete_image(html, meta, rng, params, exclude):
    """M8 (extension): delete one visible <img>. M1 covers text-block
    omission; this covers the non-text object-omission axis (G4 atomic
    blocks)."""
    cands = meta.candidates(
        lambda e: e["tag"] == "IMG" and e["paintable"] and _area(e) >= 16,
        exclude)
    if not cands:
        return None
    t = _pick(rng, cands)
    s = _soup(html)
    tag = _el(s, t["id"])
    if tag is None:
        return None
    tag.decompose()
    return MutationOutcome(str(s), (t["id"],), detail=f"deleted {_preview(t)}")


# ===========================================================================
# SAFE operators (S1-S6 from the protocol)
# ===========================================================================
def s1_strip_comments(html, meta, rng, params, exclude):
    s = _soup(html)
    found = 0
    for cm in s.find_all(string=lambda t: isinstance(t, Comment)):
        cm.extract()
        found += 1
    if not found:
        return None
    return MutationOutcome(str(s), (), detail=f"stripped {found} comments")


def s2_reorder_attributes(html, meta, rng, params, exclude):
    s = _soup(html)
    n = 0
    for t in s.find_all(True):
        if len(t.attrs) > 1:
            t.attrs = dict(reversed(list(t.attrs.items())))
            n += 1
    if not n:
        return None
    return MutationOutcome(str(s), (), detail=f"reordered attrs on {n} tags")


_TAG_SEG = re.compile(r"(<[^>]*>)")
_UNQUOTABLE = re.compile(r'([a-zA-Z_][\w:.-]*)="([A-Za-z0-9._:\-]+)"')
_RAW_OPEN = re.compile(r"^<\s*(script|style)\b", re.I)
_RAW_CLOSE = re.compile(r"^<\s*/\s*(script|style)\s*>", re.I)


def s3_requote_attributes(html, meta, rng, params, exclude):
    """S3: unquote attribute values that are spec-valid unquoted (no spaces,
    quotes, =, <, >, backtick). Operates only inside tag tokens, skipping
    <script>/<style> raw text so inline JS/CSS strings are never touched."""
    parts = _TAG_SEG.split(html)
    in_raw = None
    changed = 0
    for i, seg in enumerate(parts):
        if seg.startswith("<"):
            if in_raw:
                if _RAW_CLOSE.match(seg):
                    in_raw = None
                continue
            if _RAW_OPEN.match(seg):
                in_raw = _RAW_OPEN.match(seg).group(1).lower()
            if seg.startswith(("<!--", "<!")):
                continue
            new = _UNQUOTABLE.sub(lambda m: f"{m.group(1)}={m.group(2)}", seg)
            if new != seg:
                parts[i] = new
                changed += 1
    if not changed:
        return None
    return MutationOutcome("".join(parts), (),
                           detail=f"unquoted attrs in {changed} tags")


def s4_collapse_neutral_wrapper(html, meta, rng, params, exclude):
    """S4: unwrap one div/span whose COMPUTED style is provably neutral
    (zero box, transparent, static, unfiltered, not a flex/grid item,
    default display). This is exactly Level 2's collapse rule; the render
    verification is the arbiter of 'provably' -- any pixel change here is a
    reportable finding, not a calibration point (Stage 4.3's
    margin-collapse honesty, measured)."""
    def neutral(e):
        bg = parse_css_color(e["bg"])
        want_display = "block" if e["tag"] == "DIV" else "inline"
        return (e["tag"] in ("DIV", "SPAN") and not e["displayNone"]
                and e["nChildNodes"] >= 1
                and sum(e["pad"]) == 0 and sum(e["mar"]) == 0
                and sum(e["borderW"]) == 0
                and (bg is None or bg[3] <= 0.01)
                and e["position"] == "static" and e["opacity"] >= 0.999
                and not e["transformed"] and e["overflowVisible"]
                and e["display"] == want_display
                and e["parentDisplay"] not in
                ("flex", "grid", "inline-flex", "inline-grid"))
    cands = meta.candidates(neutral, exclude)
    if not cands:
        return None
    t = _pick(rng, cands)
    s = _soup(html)
    tag = _el(s, t["id"])
    if tag is None:
        return None
    tag.unwrap()
    return MutationOutcome(str(s), (t["id"],),
                           detail=f"collapsed neutral {_preview(t)}")


def s5_remove_display_none(html, meta, rng, params, exclude):
    """S5: remove one display:none SUBTREE ROOT, judged by COMPUTED style
    (v1 grepped inline style= only, missing every stylesheet-hidden
    subtree)."""
    def hidden_root(e):
        parent = meta.els.get(e["parentId"])
        return (e["displayNone"] and e["tag"] not in _NEVER_TOUCH
                and not (parent and parent["displayNone"]))
    cands = meta.candidates(hidden_root, exclude)
    if not cands:
        return None
    t = _pick(rng, cands)
    s = _soup(html)
    tag = _el(s, t["id"])
    if tag is None:
        return None
    tag.decompose()
    return MutationOutcome(str(s), (t["id"],),
                           detail=f"removed display:none subtree {_preview(t)}")


def s6_reminify(html, meta, rng, params, exclude):
    """S6: spec-aware re-minification via minify-html (knows inter-inline
    whitespace is significant, protects <pre>/<textarea>). The SAFE
    counterpart of the X1 probe below."""
    try:
        import minify_html
    except Exception:
        return None
    try:
        out = minify_html.minify(html, minify_css=True, keep_closing_tags=True,
                                 keep_html_and_head_opening_tags=True)
    except Exception:
        return None
    if out == html:
        return None
    return MutationOutcome(out, (), detail="minify-html pass")


# ===========================================================================
# PROBE: the old pipeline's whitespace regex, on trial
# ===========================================================================
def x1_legacy_whitespace_regex(html, meta, rng, params, exclude):
    """X1 (probe): the historical re.sub(r'>\\s+<','><') collapse the old
    Level 1 shipped, which the protocol flags as unsafe on inline runs.
    Intent 'probe': excluded from calibration; its verified break-rate is
    the quantitative justification for replacing it with minify-html
    (protocol Stage 3, item 5)."""
    out = re.sub(r">\s+<", "><", html)
    if out == html:
        return None
    return MutationOutcome(out, (), detail="legacy >\\s+< collapse")


# ===========================================================================
# Registry
# ===========================================================================
def _v(*pairs):
    return tuple((k, tuple(p.items())) for k, p in pairs)


REGISTRY: tuple = (
    MutationSpec("m1_delete_text_block", "breaking", "structure",
                 m1_delete_text_block, protocol_id="M1"),
    MutationSpec("m2_corrupt_text", "breaking", "text",
                 m2_corrupt_text, protocol_id="M2"),
    MutationSpec("m3_shift_element", "breaking", "geometry", m3_shift_element,
                 _v(("px8", {"px": 8}), ("px16", {"px": 16}), ("px32", {"px": 32})),
                 protocol_id="M3"),
    MutationSpec("m4_recolor_element", "breaking", "color", m4_recolor_element,
                 _v(("de2", {"target_de": 2}), ("de5", {"target_de": 5}),
                    ("de10", {"target_de": 10})),
                 protocol_id="M4"),
    MutationSpec("m5_fontsize_bump", "breaking", "typography", m5_fontsize_bump,
                 _v(("minus2", {"delta": -2}), ("plus2", {"delta": 2})),
                 protocol_id="M5"),
    MutationSpec("m6_remove_loadbearing_wrapper", "breaking", "structure",
                 m6_remove_loadbearing_wrapper, protocol_id="M6"),
    MutationSpec("m7_swap_siblings", "breaking", "structure",
                 m7_swap_siblings, protocol_id="M7"),
    MutationSpec("m8_delete_image", "breaking", "structure",
                 m8_delete_image, protocol_id="M8", extension=True),
    MutationSpec("s1_strip_comments", "safe", "source", s1_strip_comments,
                 protocol_id="S1"),
    MutationSpec("s2_reorder_attributes", "safe", "source",
                 s2_reorder_attributes, protocol_id="S2"),
    MutationSpec("s3_requote_attributes", "safe", "source",
                 s3_requote_attributes, protocol_id="S3"),
    MutationSpec("s4_collapse_neutral_wrapper", "safe", "structure",
                 s4_collapse_neutral_wrapper, protocol_id="S4"),
    MutationSpec("s5_remove_display_none", "safe", "structure",
                 s5_remove_display_none, protocol_id="S5"),
    MutationSpec("s6_reminify", "safe", "source", s6_reminify,
                 protocol_id="S6"),
    MutationSpec("x1_legacy_whitespace_regex", "probe", "source",
                 x1_legacy_whitespace_regex, protocol_id="X1", extension=True),
)

BY_NAME = {sp.name: sp for sp in REGISTRY}


def apply_mutation(name: str, html: str, meta, rng, variant_key: str = "default",
                   exclude_ids=frozenset()):
    """Apply one registered mutation variant. Returns MutationOutcome|None."""
    spec = BY_NAME[name]
    params = dict(spec.variants).get(variant_key)
    if params is None:
        raise KeyError(f"{name} has no variant '{variant_key}'")
    return spec.fn(html, meta, rng, params, set(exclude_ids))
