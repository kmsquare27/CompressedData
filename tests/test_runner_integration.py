"""Integration test: drive the FULL step-03 runner loop (stamp -> annotate ->
mutate with dud-resampling -> pixel-verify -> classify -> evaluate through
the production gate -> CSV -> resume) against the deterministic fake render
engine, then run step-04 calibration on the produced CSV.
Run from the repo root: python tests/test_runner_integration.py
"""
from __future__ import annotations

import argparse
import importlib
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src" / "pipeline"))
sys.path.insert(0, str(ROOT / "tests"))

import pandas as pd  # noqa: E402

from fake_render import FakeHarness  # noqa: E402
from src.compare.gate import TokenCounter, load_config  # noqa: E402

runner = importlib.import_module("03_run_stress_test")

PAGES = {
    "fk00000": """
<html><head><title>t</title></head><body>
<!-- a comment to strip -->
<h1 style="font-size:24px">Welcome To The Fake Page</h1>
<p>The quick brown fox jumps over the lazy dog near the river bank today.</p>
<p>Another paragraph with plenty of characters for corruption testing here.</p>
<div style="padding:16px;background-color:rgb(37,99,235)">
  <p>Text inside the load-bearing padded wrapper block element.</p>
</div>
<section><p>Alpha section body text content block one two three.</p></section>
<section><p>Beta section body text content block four five six.</p></section>
<img src="x.png" alt="pic">
<div style="display:none"><p>hidden machinery text</p></div>
<div><p class="k" title="v w" data-x="q">Neutral wrapper child text content.</p></div>
<div style="background-color:rgb(200,50,50)"></div>
</body></html>
""",
    "fk00001": """
<html><body>
<h2>Second Page Heading Line</h2>
<p>Alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo.</p>
<div style="padding:8px;background-color:rgb(16,185,129)">
  <p>Wrapped content sits here with some words to draw boxes.</p>
</div>
<section><p>Left column words words words words words.</p></section>
<section><p>Right column words words words words words.</p></section>
<img src="y.png">
<div style="display:none">nope</div>
<div><span style="font-size:14px">inline span text run here</span></div>
</body></html>
""",
    "fk00002": """
<html><body>
<p>Solo paragraph page with a fair amount of text to mutate safely.</p>
<p style="margin-left:12px">Indented paragraph line with more content words.</p>
<div style="background-color:rgb(250,204,21)"></div>
<section><p>Sec one content line.</p></section>
<section><p>Sec two content line.</p></section>
<img src="z.png">
</body></html>
""",
}

PASS = 0


def check(cond, msg):
    global PASS
    assert cond, msg
    PASS += 1
    print(f"  ok  {msg}")


def main():
    tmp = ROOT / "tests" / "_it_tmp"
    if tmp.exists():
        shutil.rmtree(tmp)
    (tmp / "data" / "splits").mkdir(parents=True)
    (tmp / "data" / "raw_webcode2m").mkdir(parents=True)
    (tmp / "config").mkdir(parents=True)
    shutil.copy(ROOT / "config" / "gate_config.yaml", tmp / "config")
    rows = []
    for pid, html in PAGES.items():
        p = tmp / "data" / "raw_webcode2m" / f"{pid}.html"
        p.write_text(html, encoding="utf-8")
        rows.append({"page_id": pid, "source": "webcode2m",
                     "html_path": str(p), "png_path": "", "html_chars": len(html)})
    pd.DataFrame(rows).to_csv(
        tmp / "data" / "splits" / "pilot_webcode2m_manifest.csv", index=False)

    runner.ROOT = tmp   # redirect the runner's world into the sandbox
    args = argparse.Namespace(k=3, ops=None, max_attempts=3, no_resume=False)
    cfg = load_config(tmp / "config" / "gate_config.yaml")
    tk = TokenCounter.get()

    h = FakeHarness()
    out_csv = runner.run_source("webcode2m", args, h, cfg, tk)
    runner.summarize(out_csv, "webcode2m")
    df = pd.read_csv(out_csv)

    print("\n[integration checks]")
    from src.stress import mutations as M
    per_page = sum(len(sp.variants) for sp in M.REGISTRY)
    check(out_csv.exists() and len(df) == 3 * per_page,
          f"one row per page x mutation-variant ({len(df)} == 3x{per_page})")
    # Originals render exactly once per page (cached in art_o); every mutant
    # *attempt* renders once (dud resamples re-render), na rows never render.
    check(h.n_renders == 3 + int(df["attempts"].fillna(0).sum()),
          "original rendered ONCE per page; one render per mutant attempt (cache works)")

    ok = df[df["status"] == "ok"]
    calib = ok[ok["calib_include"] == True]  # noqa: E712
    brk = calib[calib["verified_label"] == 1]
    safe = calib[calib["intent"] == "safe"]
    tol = calib[calib["intent"] == "tolerable"]
    check(len(brk) and len(safe), "both verified classes produced")
    check((~brk["accepted_visual"].astype(bool)).mean() > 0.5,
          "provisional visual gates reject most verified-breaking mutants")
    check(safe["accepted_visual"].astype(bool).all(),
          "ALL verified-safe mutants accepted by the visual gates")
    check((safe["n_diff_pixels"] == 0).all(),
          "verified-safe means pixel-identical under the (fake) harness")
    check((brk["n_diff_pixels"] > 0).all(),
          "verified-breaking means pixels actually changed")
    # The sub-JND class is the point of the tolerable intent: negatives that
    # DO change pixels, hence the only samples able to bound a threshold from
    # above (pixel-identical negatives are accepted at any threshold).
    check((tol["verified_label"] == 0).all() if len(tol) else True,
          "sub-JND mutants are labelled NEGATIVE")
    check((tol["n_diff_pixels"] > 0).all() if len(tol) else True,
          "calibrated sub-JND mutants changed pixels (they can constrain)")
    noop = df[df["status"] == "tolerable_noop"]
    check(((noop["calib_include"] != True).all() if len(noop) else True),  # noqa: E712
          "sub-JND mutants that changed nothing are excluded, not padded in")

    m4 = brk[brk["protocol_id"] == "M4"]
    if len(m4):
        err = (pd.to_numeric(m4["deltae_max"])
               - pd.to_numeric(m4["severity_achieved"])).abs()
        check((err < 1.5).all(),
              "G5 deltae_max tracks the injected dE00 severity "
              f"(max err {err.max():.2f})")
    m3 = brk[(brk["protocol_id"] == "M3")]
    if len(m3):
        ok3 = (pd.to_numeric(m3["center_shift_max"])
               >= pd.to_numeric(m3["severity_nominal"]) - 1).all()
        check(ok3, "G4 center_shift reflects the injected px shift")
    m1 = brk[brk["protocol_id"] == "M1"]
    # Deletion usually orphans a block, but reflow can keep counts equal
    # (pairwise matching then degrades IoU instead) -- G4 must reject either way.
    check((pd.to_numeric(m1["n_unmatched"]) > 0).any(),
          "M1 deletion produces unmatched blocks on at least one page")
    check((~m1["g4_blocks"].astype(bool)).all(),
          "M1 deletion always fails G4 (unmatched blocks or IoU collapse)")
    g1_up = brk[pd.to_numeric(brk["tokens_comp"])
                > pd.to_numeric(brk["tokens_orig"])]
    check(len(g1_up) > 0 and (~g1_up["g1_tokens"].astype(bool)).all(),
          "style-adding mutants DO fail G1 -- which is why the verdict "
          "excludes it")

    x1 = df[(df["mutation"] == "x1_legacy_whitespace_regex")]
    check(len(x1) == 3 and (x1["calib_include"] == False).all(),  # noqa: E712
          "X1 probe present and excluded from calibration")

    n_before = len(df)
    h2 = FakeHarness()
    runner.run_source("webcode2m", args, h2, cfg, tk)
    df2 = pd.read_csv(out_csv)
    check(len(df2) == n_before and h2.n_renders == 0,
          "resume: second run renders nothing, appends nothing")

    # determinism across a fresh full run
    shutil.move(str(out_csv), str(out_csv) + ".first")
    h3 = FakeHarness()
    runner.run_source("webcode2m", args, h3, cfg, tk)
    a = pd.read_csv(str(out_csv) + ".first").sort_values("sample_id")
    b = pd.read_csv(out_csv).sort_values("sample_id")
    same = (a["detail"].fillna("").tolist() == b["detail"].fillna("").tolist()
            and a["target_ids"].fillna("").tolist() == b["target_ids"].fillna("").tolist())
    check(same, "byte-level reproducibility: fresh run picks identical targets")

    # step 04 runs on real-machinery output
    calmod = importlib.import_module("04_calibrate_gate")
    calmod.ROOT = tmp
    sys.argv = ["04", "--source", "webcode2m"]
    calmod.main()
    check((tmp / "reports" / "csv" / "gate_calibration.csv").exists()
          and (tmp / "config" / "gate_config.calibrated.yaml").exists()
          and (tmp / "reports" / "stress_decisions.md").exists(),
          "step 04 calibrates the runner's real output end-to-end")

    print(f"\nALL {PASS} INTEGRATION CHECKS PASSED")
    shutil.rmtree(tmp)


if __name__ == "__main__":
    main()