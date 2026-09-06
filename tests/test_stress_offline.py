"""Offline tests for the stress-test subsystem (no browser required).

Covers: operator correctness on fabricated annotations, the dE00-targeted
recolor search, determinism, label classification, pixel verification,
G1-exclusion in the visual verdict, and a synthetic end-to-end run of the
step-04 calibration. Run from the repo root:  python tests/test_stress_offline.py
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
from bs4 import BeautifulSoup  # noqa: E402

from src.stress import annotate  # noqa: E402
from src.stress import mutations as M  # noqa: E402
from src.stress.annotate import PageMeta  # noqa: E402

PASS = 0


def check(cond, msg):
    global PASS
    assert cond, msg
    PASS += 1
    print(f"  ok  {msg}")


def E(id, tag="DIV", **kw):
    base = dict(id=id, tag=tag, parentId=-1, x=0, y=0, w=100, h=40,
                display="block", position="static", visibility="visible",
                opacity=1.0, paintable=True, displayNone=False,
                leafTextLen=0, textLen=0, bg="rgba(0, 0, 0, 0)",
                fontSize=16.0, pad=[0, 0, 0, 0], mar=[0, 0, 0, 0],
                borderW=[0, 0, 0, 0], transformed=False, overflowVisible=True,
                parentDisplay="block", nChildElems=0, nChildNodes=1)
    base.update(kw)
    return base


def META(*els):
    return PageMeta({"els": list(els), "docW": 1280, "docH": 800})


def dom_equal(a: str, b: str) -> bool:
    import lxml.html as LH

    def norm(n):
        return (n.tag, sorted((n.attrib or {}).items()),
                (n.text or "").strip(), [norm(c) for c in n])
    return norm(LH.fromstring(a)) == norm(LH.fromstring(b))


# ---------------------------------------------------------------------- stamping
def test_stamping():
    print("[stamping]")
    html = "<html><head><style>p{color:red}</style></head><body><div><p>hi</p></div></body></html>"
    stamped, n = annotate.stamp_ids(html)
    check(n == 3, "stamps body + 2 descendants")
    s = BeautifulSoup(stamped, "lxml")
    check(s.body[annotate.STAMP_ATTR] == "0", "body gets id 0 (document order)")
    check(s.find("p")[annotate.STAMP_ATTR] == "2", "ids follow document order")
    check(not annotate.data_attr_selector_hazard(html), "no [data- selector -> no hazard")
    check(annotate.data_attr_selector_hazard(
        "<html><head><style>[data-x]{color:red}</style></head><body></body></html>"),
        "[data- attribute selector detected as hazard")


# ---------------------------------------------------------------------- operators
def _stamped(body_inner: str) -> str:
    stamped, _ = annotate.stamp_ids(f"<html><body>{body_inner}</body></html>")
    return stamped


def test_m1_targets_only_visible():
    print("[M1] annotation-driven targeting")
    html = _stamped('<p>visible text</p><p style="display:none">ghost text</p>')
    # ids: 0=body, 1=first p, 2=second p. Only id 1 is paintable.
    meta = META(E(0, "BODY", nChildElems=2),
                E(1, "P", leafTextLen=12, textLen=12),
                E(2, "P", leafTextLen=10, textLen=10, paintable=False,
                  displayNone=True))
    out = M.apply_mutation("m1_delete_text_block", html, meta,
                           random.Random(0))
    s = BeautifulSoup(out.html, "lxml")
    check(out.target_ids == (1,), "picks the paintable block, never the ghost")
    check("ghost text" in out.html and "visible text" not in s.get_text(),
          "deletes exactly the visible block")
    # exclude the only candidate -> NA, the dud-resampling hook
    check(M.apply_mutation("m1_delete_text_block", html, meta,
                           random.Random(0), exclude_ids={1}) is None,
          "exclude_ids removes the target pool (dud resampling works)")


def test_m2_preserves_children():
    print("[M2] pure text change (v1 deleted child elements)")
    html = _stamped('<p>Read the documentation <a href="#">right here</a> before continuing today</p>')
    meta = META(E(0, "BODY"), E(1, "P", leafTextLen=40, textLen=50),
                E(2, "A", leafTextLen=10, textLen=10))
    rng = random.Random(1)
    out = M.apply_mutation("m2_corrupt_text", html, meta, rng)
    s = BeautifulSoup(out.html, "lxml")
    check(s.find("a") is not None and s.find("a").get_text() == "right here",
          "child <a> survives intact")
    orig_p = BeautifulSoup(html, "lxml").find("p").get_text()
    new_p = s.find("p").get_text()
    check(orig_p != new_p and len(orig_p) == len(new_p),
          "direct text changed, length preserved")


def test_m3_m5_relative_edits():
    print("[M3/M5] relative to computed values")
    html = _stamped("<p>some paragraph text</p>")
    meta = META(E(0, "BODY"), E(1, "P", leafTextLen=19, textLen=19,
                                mar=[0, 0, 0, 12], fontSize=23.0))
    o3 = M.apply_mutation("m3_shift_element", html, meta, random.Random(2), "px16")
    check("margin-left:28px !important" in o3.html,
          "M3 adds 16 to the COMPUTED 12px margin")
    check(o3.severity_nominal == 16.0, "M3 severity recorded")
    o5 = M.apply_mutation("m5_fontsize_bump", html, meta, random.Random(2), "plus2")
    check("font-size:25px !important" in o5.html,
          "M5 is relative: 23 -> 25px (v1's absolute 23px would be a no-op)")


def test_m4_delta_e_targeting():
    print("[M4] CIEDE2000-targeted recolor")
    for base in [(255, 255, 255), (128, 128, 128), (37, 99, 235), (0, 0, 0)]:
        for target in (2, 5, 10):
            got = M.color_at_delta_e(base, target)
            check(got is not None and abs(got[1] - target) <= max(0.5, 0.25 * target),
                  f"base {base} reaches dE00~{target} (achieved {got[1] if got else None})")
    html = _stamped('<div style="background:#2563eb">block</div>')
    meta = META(E(0, "BODY"), E(1, "DIV", bg="rgb(37, 99, 235)",
                                leafTextLen=5, textLen=5, w=200, h=80))
    out = M.apply_mutation("m4_recolor_element", html, meta,
                           random.Random(3), "de5")
    check(out is not None and "background-color:rgb(" in out.html
          and out.severity_achieved is not None,
          "M4 injects the found color and records achieved severity")
    # transparent backgrounds are not candidates
    meta_t = META(E(0, "BODY"), E(1, "DIV", bg="rgba(0, 0, 0, 0)", w=200, h=80))
    check(M.apply_mutation("m4_recolor_element", html, meta_t,
                           random.Random(3), "de5") is None,
          "transparent-bg element rejected as recolor target")


def test_m6_computed_not_inline():
    print("[M6] finds stylesheet-styled wrappers (v1 needed inline style=)")
    html = _stamped('<div class="card"><p>content</p></div>')
    meta = META(E(0, "BODY", nChildElems=1),
                E(1, "DIV", nChildElems=1, pad=[16, 16, 16, 16]),  # via CSS class
                E(2, "P", leafTextLen=7, textLen=7))
    out = M.apply_mutation("m6_remove_loadbearing_wrapper", html, meta,
                           random.Random(4))
    s = BeautifulSoup(out.html, "lxml")
    check(out.target_ids == (1,) and s.find("div") is None
          and s.find("p") is not None,
          "unwraps the computed-padded wrapper, keeps children")


def test_m7_swap():
    print("[M7] sibling swap")
    html = _stamped("<section><h2>Alpha</h2></section><section><h2>Beta</h2></section>")
    meta = META(E(0, "BODY", nChildElems=2),
                E(1, "SECTION", parentId=0, w=400, h=100, nChildElems=1),
                E(2, "H2", parentId=1, leafTextLen=5, textLen=5),
                E(3, "SECTION", parentId=0, w=400, h=100, nChildElems=1),
                E(4, "H2", parentId=3, leafTextLen=4, textLen=4))
    out = M.apply_mutation("m7_swap_siblings", html, meta, random.Random(5))
    body = BeautifulSoup(out.html, "lxml").body
    secs = body.find_all("section")
    check(sorted(out.target_ids) == [1, 3], "targets the two siblings")
    check(secs[0].get_text(strip=True) == "Beta"
          and secs[1].get_text(strip=True) == "Alpha", "order swapped")
    check(len(body.find_all("template")) == 0, "no placeholder left behind")


def test_safe_ops():
    print("[S3/S4/S5] safe operators")
    html = _stamped('<script>var s = \'<a href="x">\';</script>'
                    '<div class="wrap"><a href="page.html" title="a b">link</a></div>')
    o3 = M.apply_mutation("s3_requote_attributes", html, META(E(0, "BODY")),
                          random.Random(6))
    check('var s = \'<a href="x">\';' in o3.html,
          "S3 never touches <script> raw text")
    check('title="a b"' in o3.html, "S3 keeps quotes on unsafe values")
    check("class=wrap" in o3.html and "href=page.html" in o3.html,
          "S3 unquotes spec-safe values")
    check(dom_equal(html, o3.html), "S3 output parses to the identical DOM")

    # S4 neutrality: flex-item wrapper must NOT be a candidate
    html4 = _stamped("<div><p>x</p></div>")
    neutral = E(1, "DIV", nChildElems=1, nChildNodes=1)
    flexkid = E(1, "DIV", nChildElems=1, nChildNodes=1, parentDisplay="flex")
    check(M.apply_mutation("s4_collapse_neutral_wrapper", html4,
                           META(E(0, "BODY"), neutral, E(2, "P", leafTextLen=1)),
                           random.Random(7)) is not None,
          "S4 collapses a computed-neutral div")
    check(M.apply_mutation("s4_collapse_neutral_wrapper", html4,
                           META(E(0, "BODY"), flexkid, E(2, "P", leafTextLen=1)),
                           random.Random(7)) is None,
          "S4 refuses a flex-item wrapper (removal would reflow siblings)")

    # S5 works from COMPUTED display, no inline style needed
    html5 = _stamped('<div class="hidden-by-css">secret</div><p>shown</p>')
    meta5 = META(E(0, "BODY"), E(1, "DIV", displayNone=True, paintable=False,
                                 textLen=6), E(2, "P", leafTextLen=5))
    o5 = M.apply_mutation("s5_remove_display_none", html5, meta5,
                          random.Random(8))
    check("secret" not in o5.html and "shown" in o5.html,
          "S5 removes the stylesheet-hidden subtree")


def test_determinism():
    print("[determinism] stable seeds reproduce mutants byte-for-byte")
    html = _stamped("<p>alpha</p><p>beta</p><p>gamma</p>")
    meta = META(E(0, "BODY"), *[E(i, "P", leafTextLen=5, textLen=5)
                                for i in (1, 2, 3)])
    a = M.apply_mutation("m1_delete_text_block", html, meta, random.Random(42))
    b = M.apply_mutation("m1_delete_text_block", html, meta, random.Random(42))
    check(a.html == b.html and a.target_ids == b.target_ids,
          "same seed -> identical mutant")
    picks = {M.apply_mutation("m1_delete_text_block", html, meta,
                              random.Random(s)).target_ids[0]
             for s in range(12)}
    check(len(picks) > 1, "different seeds explore different targets")


def test_classify_and_pixels():
    print("[verification] label classification + pixel ground truth")
    sys.path.insert(0, str(ROOT / "src" / "pipeline"))
    import importlib
    runner = importlib.import_module("03_run_stress_test")
    check(runner.classify("breaking", True) == ("ok", 1, True),
          "breaking + changed -> calibration positive")
    check(runner.classify("breaking", False) == ("dud", 0, False),
          "breaking + unchanged -> DUD, excluded (never a gate miss)")
    check(runner.classify("safe", False) == ("ok", 0, True),
          "safe + unchanged -> calibration negative")
    check(runner.classify("safe", True) == ("unsafe_safe", 1, False),
          "safe + changed -> finding, excluded from calibration")
    check(runner.classify("probe", True)[2] is False,
          "probe never enters calibration")
    check(runner.stable_seed("p1", "m1", "default", 1)
          == runner.stable_seed("p1", "m1", "default", 1)
          != runner.stable_seed("p1", "m1", "default", 2),
          "stable_seed is deterministic and attempt-sensitive")

    a = np.zeros((10, 10, 3), np.uint8)
    b = a.copy()
    check(annotate.pixel_stats(a, b)["n_diff_pixels"] == 0, "identical -> 0 diff")
    b2 = a.copy(); b2[3, 4] = 255
    st = annotate.pixel_stats(a, b2)
    check(st["n_diff_pixels"] == 1 and st["diff_bbox"] == "4,3,4,3",
          "single-pixel diff located")
    tall = np.zeros((20, 10, 3), np.uint8)
    st2 = annotate.pixel_stats(tall, a)
    check(st2["n_diff_pixels"] == 100 and st2["shape_mismatch"],
          "lost lower half counts fully as different (never resized)")


def test_visual_verdict_ignores_g1():
    print("[G1 exclusion] visual verdict cannot be faked by token direction")
    from src.compare.gate import PageArtifacts, evaluate_pair, visual_accepted
    img = np.full((800, 1280, 3), 255, np.uint8)
    box = [{"text": "hello world", "tag": "P", "x": 10, "y": 10, "w": 200, "h": 20}]
    lay = {"boxes": box, "docW": 1280, "docH": 800, "text": "hello world"}
    a = PageArtifacts("o.html", "o.png", "<p>hello world</p>", 10, lay, img)
    # identical render, MORE tokens (a style-adding mutant): G1 fails,
    # visual gates all pass -> visual verdict is ACCEPT (correct: no pixel change)
    b = PageArtifacts("c.html", "c.png",
                      '<p style="x:y">hello world</p>', 14, lay, img)
    R = evaluate_pair(a, b, {"height_tol_px": 2, "text_ratio_min": 0.995,
                             "iou_mean_min": 0.95, "center_shift_max_px": 5,
                             "deltae_p95_max": 2.0, "deltae_max": 5.0,
                             "require_token_reduction": True,
                             "use_lpips": False, "lpips_max": 0.03})
    check(R["g1_tokens"] is False and R["accepted"] is False,
          "full gate still enforces G1 for production")
    check(visual_accepted(R) is True,
          "visual verdict ignores G1 (v1's contamination bug is closed)")
    # and a real text change trips G3 in the visual verdict. Corrupted text
    # necessarily changes pixels, so the mutant gets its OWN render here --
    # reusing `img` would describe a page whose text changed with an
    # identical raster, which cannot happen and which the gate's soundness
    # rule (identical pixels => visually equivalent) correctly accepts.
    img2 = img.copy()
    img2[100:140, 60:400] = 0
    lay2 = {"boxes": [dict(box[0], text="hello mars")], "docW": 1280,
            "docH": 800, "text": "hello mars"}
    c = PageArtifacts("c2.html", "c2.png", "<p>hello mars</p>", 9, lay2, img2)
    R2 = evaluate_pair(a, c, {"height_tol_px": 2, "text_ratio_min": 0.995,
                              "iou_mean_min": 0.95, "center_shift_max_px": 5,
                              "deltae_p95_max": 2.0, "deltae_max": 5.0,
                              "require_token_reduction": True,
                              "use_lpips": False, "lpips_max": 0.03})
    check(R2["g3_text"] is False and visual_accepted(R2) is False,
          "text corruption is caught by G3, not by G1")


def test_probe_x1():
    print("[X1] the legacy regex probe fires")
    html = "<html><body>\n  <div>\n    <span>a</span> <span>b</span>\n  </div>\n</body></html>"
    out = M.apply_mutation("x1_legacy_whitespace_regex", html,
                           META(E(0, "BODY")), random.Random(0))
    check(out is not None and "> <" not in out.html.replace("</span> <span", ""),
          "X1 collapses inter-tag whitespace (incl. the unsafe inline run)")


def run_all():
    for fn in [test_stamping, test_m1_targets_only_visible,
               test_m2_preserves_children, test_m3_m5_relative_edits,
               test_m4_delta_e_targeting, test_m6_computed_not_inline,
               test_m7_swap, test_safe_ops, test_determinism,
               test_classify_and_pixels, test_visual_verdict_ignores_g1,
               test_probe_x1, test_subjnd_operators,
               test_tolerable_classification]:
        fn()
    print(f"\nALL {PASS} CHECKS PASSED")



# ---------------------------------------------------------------------------
# T1/T2 -- sub-JND negatives (the only samples that can constrain a threshold)
# ---------------------------------------------------------------------------
def test_subjnd_operators():
    print("[T1/T2] sub-JND tolerable class")
    for base in [(255, 255, 255), (243, 244, 246), (128, 128, 128),
                 (37, 99, 235), (0, 0, 0), (17, 24, 39)]:
        got = M.nearest_subjnd_color(base, 0.5, cap=1.0)
        check(got is not None and 0.0 < got[1] < 1.0,
              f"base {base} admits a strictly sub-JND colour "
              f"(dE00={got[1] if got else '-'})")

    t1, t2 = M.BY_NAME["t1_recolor_subjnd"], M.BY_NAME["t2_shift_subpixel"]
    check(t1.intent == "tolerable" and t2.intent == "tolerable",
          "T1/T2 declare intent=tolerable")
    check(t1.max_diff_frac == 0.0,
          "T1 needs no integrity screen (sub-JND bound holds by construction)")
    check(t2.max_diff_frac > 0.0,
          "T2 carries a diff_frac screen (0.5px may trigger a line re-wrap)")

    html = ('<html><body><div data-mut-id="0">'
            '<div data-mut-id="1">hello world</div></div></body></html>')
    meta = META(E(0, "BODY", nChildElems=1),
                E(1, "DIV", bg="rgb(255, 255, 255)", w=400, h=40,
                  leafTextLen=11, textLen=11))
    out = M.apply_mutation("t1_recolor_subjnd", html, meta,
                           random.Random(0), "de0_5", set())
    check(out is not None and 0.0 < out.severity_achieved < 1.0,
          f"T1 emits a sub-JND recolour "
          f"(dE00={out.severity_achieved if out else '-'})")
    out2 = M.apply_mutation("t2_shift_subpixel", html, meta,
                            random.Random(0), "px0_5", set())
    check(out2 is not None and "0.5px" in out2.html,
          "T2 emits a half-pixel margin shift")


def test_tolerable_classification():
    print("[classify] tolerable is a NEGATIVE that must change pixels")
    sys.path.insert(0, str(ROOT / "src" / "pipeline"))
    import importlib
    runner = importlib.import_module("03_run_stress_test")
    check(runner.classify("tolerable", True) == ("ok", 0, True),
          "tolerable + changed -> calibration NEGATIVE (constrains thresholds)")
    check(runner.classify("tolerable", False) == ("tolerable_noop", 0, False),
          "tolerable + unchanged -> excluded (cannot constrain anything)")
    check(runner.classify("safe", False) == ("ok", 0, True),
          "safe class unchanged by the addition")
    check(runner.classify("breaking", True) == ("ok", 1, True),
          "breaking class unchanged by the addition")


if __name__ == "__main__":
    run_all()