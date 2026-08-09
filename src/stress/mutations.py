"""Mutation library for the gate stress test.

BREAKING mutations (label=1) change what the page looks like -> a sound gate
must REJECT them. SAFE mutations (label=0) are visually neutral -> a sound
gate must ACCEPT them. Detection rate on breaking vs false-rejection rate on
safe, per metric, is how thresholds get calibrated (and how you prove SSIM
alone is insufficient).

Each mutation: fn(html:str, rng:random.Random) -> str|None (None = not
applicable to this page; the runner skips it).
"""
from __future__ import annotations

import random
import re

from bs4 import BeautifulSoup, Comment

TEXT_TAGS = ["p", "h1", "h2", "h3", "h4", "h5", "h6", "li", "a", "span",
             "button", "td", "label"]


def _soup(html: str) -> BeautifulSoup:
    return BeautifulSoup(html, "lxml")


def _text_candidates(soup):
    return [t for t in soup.find_all(TEXT_TAGS) if t.get_text(strip=True)]


def _append_style(tag, css: str) -> None:
    prev = tag.get("style", "")
    tag["style"] = (prev.rstrip("; ") + ";" if prev else "") + css


# --------------------------- BREAKING (label = 1) ---------------------------
def m1_delete_text_block(html, rng):
    s = _soup(html); c = _text_candidates(s)
    if not c:
        return None
    rng.choice(c).decompose()
    return str(s)


def m2_corrupt_text(html, rng):
    s = _soup(html); c = _text_candidates(s)
    if not c:
        return None
    t = rng.choice(c)
    txt = t.get_text()
    chars = list(txt)
    idx = [i for i, ch in enumerate(chars) if ch.isalnum()]
    if len(idx) < 4:
        return None
    for i in rng.sample(idx, max(1, len(idx) // 3)):
        chars[i] = "x"
    t.string = "".join(chars)
    return str(s)


def m3_shift_element(html, rng, px: int = 16):
    s = _soup(html); c = _text_candidates(s) or s.find_all("div")
    if not c:
        return None
    _append_style(rng.choice(c), f"margin-left:{px}px")
    return str(s)


def m4_recolor_element(html, rng):
    s = _soup(html)
    c = s.find_all(["div", "section", "header", "footer", "body"])
    if not c:
        return None
    _append_style(rng.choice(c), "background-color:#d81b60")
    return str(s)


def m5_fontsize_bump(html, rng):
    s = _soup(html); c = _text_candidates(s)
    if not c:
        return None
    _append_style(rng.choice(c), "font-size:23px")
    return str(s)


def m6_remove_padded_wrapper(html, rng):
    s = _soup(html)
    c = [d for d in s.find_all("div")
         if "padding" in (d.get("style") or "") and d.find(True)]
    if not c:
        return None
    rng.choice(c).unwrap()  # keep children, drop the load-bearing box
    return str(s)


# ----------------------------- SAFE (label = 0) -----------------------------
def s1_strip_comments(html, rng):
    s = _soup(html)
    found = False
    for cm in s.find_all(string=lambda t: isinstance(t, Comment)):
        cm.extract(); found = True
    return str(s) if found else None


def s2_reorder_attributes(html, rng):
    s = _soup(html)
    for t in s.find_all(True):
        if len(t.attrs) > 1:
            t.attrs = dict(sorted(t.attrs.items()))
    return str(s)


def s3_remove_display_none(html, rng):
    s = _soup(html)
    hidden = [t for t in s.find_all(style=True)
              if "display:none" in t["style"].replace(" ", "")]
    if not hidden:
        return None
    for t in hidden:
        t.decompose()
    return str(s)


def s4_collapse_whitespace(html, rng):
    if "<pre" in html.lower():
        return None  # stay out of <pre> pages in the safe set
    return re.sub(r"[ \t\r\n]{2,}", " ", html)


MUTATIONS = {
    "m1_delete_text_block": (m1_delete_text_block, 1),
    "m2_corrupt_text": (m2_corrupt_text, 1),
    "m3_shift_element": (m3_shift_element, 1),
    "m4_recolor_element": (m4_recolor_element, 1),
    "m5_fontsize_bump": (m5_fontsize_bump, 1),
    "m6_remove_padded_wrapper": (m6_remove_padded_wrapper, 1),
    "s1_strip_comments": (s1_strip_comments, 0),
    "s2_reorder_attributes": (s2_reorder_attributes, 0),
    "s3_remove_display_none": (s3_remove_display_none, 0),
    "s4_collapse_whitespace": (s4_collapse_whitespace, 0),
}


def apply_mutation(name: str, html: str, seed: int):
    fn, label = MUTATIONS[name]
    return fn(html, random.Random(seed)), label
