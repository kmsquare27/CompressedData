"""Step 07 (v3) -- per-page level selection over Levels 1, 2 and 3.

Every page keeps a target; rejection only means a less-compressed target.
Selection is on ABSOLUTE TOKENS (the gate CSVs measure reduction_pct
against different bases: L2's and L3's against whatever file they started
from), so min() over token counts is the only well-defined comparison.

What changed vs v2:
  * Level 3 (outputs/level3, level3_gate_<src>.csv) joins the candidates.
  * The output is a LADDER per page -- every gate-accepted artifact sorted
    by tokens -- not just the winner, because step 08 re-validates the
    winner against the ORIGINAL render (levels were gated against their own
    base; tolerances could stack) and falls back one rung if it fails.
  * The dataset's own screenshot is kept as `dataset_png_path`; the
    training screenshot is the harness render, written by step 08. Pairing
    a compressed target with a screenshot rendered under different
    conditions (real images, remote CSS, another viewport) would pair it
    with an image the equivalence proof never saw.

Usage:
    python src/pipeline/07_select_levels.py --source webcode2m

Output: reports/csv/level_selection_<source>.csv   (+ ladder JSON column)
        data/splits/selected_<source>_manifest.csv  provisional; step 08
                                                    writes the final one
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402


def _accepted(df: pd.DataFrame) -> pd.Series:
    if "accepted" not in df.columns:
        return pd.Series(False, index=df.index)
    return df["accepted"].astype(str).str.lower().isin(["true", "1"])


def _flag(v) -> bool:
    """NaN (level missing for this page) must read as False, not True."""
    return bool(v) and not (isinstance(v, float) and pd.isna(v))


def _num(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(float("nan"), index=df.index)
    return pd.to_numeric(df[col], errors="coerce")


_BASE_NORM = {"level1": "l1", "level2": "l2", "level3": "l3"}


def load_level(path: Path, level: str) -> pd.DataFrame:
    if not path.exists():
        print(f"[07] no {path.name}; treating {level} as unavailable")
        return pd.DataFrame(columns=["page_id", f"{level}_ok", f"{level}_tokens",
                                     f"{level}_orig_tokens", f"{level}_base"])
    d = pd.read_csv(path)
    d = d.drop_duplicates("page_id", keep="last")
    # `base` is absent from the Level-1 CSV. Building the Series first keeps
    # `.where()` from receiving a scalar True (which raised ValueError).
    base_col = (d["base"].astype(str) if "base" in d.columns
                else pd.Series("original", index=d.index))
    # 06 records the raw stage name ("level1"); normalize to the "l1" label
    # the composition table below matches on, so stacked pages are counted.
    base_col = base_col.map(lambda v: _BASE_NORM.get(v, v))
    out = pd.DataFrame({
        "page_id": d["page_id"].astype(str),
        f"{level}_ok": _accepted(d),
        f"{level}_tokens": _num(d, "tokens_comp"),
        # Level 1 always measures against the true original; L2/L3 record
        # it explicitly in tokens_original (v3) or only when base=original.
        f"{level}_orig_tokens": (_num(d, "tokens_original")
                                 if "tokens_original" in d.columns
                                 else _num(d, "tokens_orig").where(base_col == "original")),
        f"{level}_base": base_col,
    })
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="webcode2m")
    args = ap.parse_args()
    src = args.source

    man = ROOT / "data" / "splits" / f"pilot_{src}_manifest.csv"
    m = pd.read_csv(man)
    m["page_id"] = m["page_id"].astype(str)
    rep = ROOT / "reports" / "csv"
    levels = ["l1", "l2", "l3"]
    dirs = {"l1": ROOT / "outputs" / "level1" / src,
            "l2": ROOT / "outputs" / "level2" / src,
            "l3": ROOT / "outputs" / "level3" / src}
    df = m
    for lv in levels:
        df = df.merge(load_level(rep / f"level{lv[-1]}_gate_{src}.csv", lv),
                      on="page_id", how="left")

    # L1a/L1b rungs: finer-grained than the single `l1` rung (whichever stage
    # 06 composed on), so step 08 can fall back to a stage even when the
    # final L1 pick and the winning stage happen to be the same file.
    l1_stage_dirs = {"l1a": ROOT / "outputs" / "level1a" / src,
                     "l1b": ROOT / "outputs" / "level1b" / src}
    l1_stage_rungs: dict[str, list] = {}
    stages_csv = rep / f"level1_stages_{src}.csv"
    if stages_csv.exists():
        st = pd.read_csv(stages_csv)
        st["page_id"] = st["page_id"].astype(str)
        st = st[st["stage"].isin(["l1a", "l1b"]) & _accepted(st)]
        for _, sr in st.iterrows():
            p = l1_stage_dirs[sr["stage"]] / f"{sr['page_id']}.html"
            if not p.exists() or pd.isna(sr.get("tokens_comp")):
                continue
            l1_stage_rungs.setdefault(sr["page_id"], []).append(
                {"level": sr["stage"], "html_path": str(p),
                 "tokens": float(sr["tokens_comp"]), "base": "original"})
    else:
        print(f"[07] no {stages_csv.name}; L1a/L1b ladder rungs unavailable")

    orig = df["l1_orig_tokens"]
    for lv in ("l2", "l3"):
        orig = orig.fillna(df[f"{lv}_orig_tokens"])

    rows = []
    for _, r in df.iterrows():
        pid = r["page_id"]
        t_orig = orig.loc[r.name]
        ladder = []
        for lv in levels:
            p = dirs[lv] / f"{pid}.html"
            if _flag(r.get(f"{lv}_ok")) and p.exists() and pd.notna(r.get(f"{lv}_tokens")):
                ladder.append({"level": lv, "html_path": str(p),
                               "tokens": float(r[f"{lv}_tokens"]),
                               "base": str(r.get(f"{lv}_base", ""))})
        ladder.extend(l1_stage_rungs.get(pid, []))
        ladder = list({c["html_path"]: c for c in ladder}.values())
        ladder.sort(key=lambda c: c["tokens"])
        ladder.append({"level": "original", "html_path": str(r["html_path"]),
                       "tokens": float(t_orig) if pd.notna(t_orig) else float("nan"),
                       "base": ""})
        top = ladder[0]
        red = (100.0 * (t_orig - top["tokens"]) / t_orig
               if pd.notna(t_orig) and pd.notna(top["tokens"]) and t_orig else 0.0)
        rows.append({
            "page_id": pid, "source": src,
            "dataset_png_path": r.get("png_path", ""),
            "html_path": top["html_path"], "level": top["level"],
            "tokens_orig": t_orig, "tokens_final": top["tokens"],
            "reduction_pct": round(float(red), 3),
            "l1_ok": _flag(r.get("l1_ok")), "l2_ok": _flag(r.get("l2_ok")),
            "l3_ok": _flag(r.get("l3_ok")),
            "l2_base": r.get("l2_base", ""), "l3_base": r.get("l3_base", ""),
            "ladder": json.dumps(ladder),
        })

    out = pd.DataFrame(rows)
    sel_csv = rep / f"level_selection_{src}.csv"
    out.to_csv(sel_csv, index=False)
    man_cols = ["page_id", "source", "dataset_png_path", "html_path", "level",
                "tokens_orig", "tokens_final", "reduction_pct"]
    man_csv = ROOT / "data" / "splits" / f"selected_{src}_manifest.csv"
    out[man_cols].to_csv(man_csv, index=False)

    # ---- report -----------------------------------------------------------
    n = len(out)
    print(f"\n[07] ---- provisional selection: {src} ({n} pages, none dropped) ----\n")
    print(f"{'selected level':<20}{'pages':>7}{'mean red.':>11}")
    for lv in ["l3", "l2", "l1", "original"]:
        sub = out[out.level == lv]
        if len(sub):
            print(f"{lv:<20}{len(sub):>7}{sub['reduction_pct'].mean():>10.2f}%")
    # The four-case composition table (L2 stacked on L1 vs L2 after L1 failed
    # are different stories and the per-level count above conflates them).
    print(f"\n{'composition':<28}{'pages':>7}{'mean red.':>11}")
    cases = [("L1 + L2 stacked", (out.l1_ok & out.l2_ok & (out.l2_base == "l1"))),
             ("L2 only (L1 rejected)", (~out.l1_ok & out.l2_ok)),
             ("L2 rejected, L1 kept", (out.l1_ok & ~out.l2_ok)),
             ("L1 kept, L2 on original", (out.l1_ok & out.l2_ok & (out.l2_base == "original"))),
             ("neither", (~out.l1_ok & ~out.l2_ok))]
    for name, mask in cases:
        sub = out[mask.fillna(False)]
        if len(sub):
            print(f"{name:<28}{len(sub):>7}{sub['reduction_pct'].mean():>10.2f}%")
    if out.l3_ok.any():
        print(f"{'L3 accepted (any base)':<28}{int(out.l3_ok.sum()):>7}")
    comp = out[out.level != "original"]
    tot_o = pd.to_numeric(out["tokens_orig"], errors="coerce").sum()
    tot_f = pd.to_numeric(out["tokens_final"], errors="coerce").sum()
    print(f"\ncompressed pages           : {len(comp)}/{n}")
    if len(comp):
        print(f"reduction on those pages   : mean {comp['reduction_pct'].mean():.2f}%  "
              f"median {comp['reduction_pct'].median():.2f}%  "
              f"p90 {comp['reduction_pct'].quantile(.9):.2f}%")
    if tot_o:
        print(f"CORPUS-LEVEL token saving  : {100.0 * (tot_o - tot_f) / tot_o:.2f}%  "
              f"({int(tot_o - tot_f):,} of {int(tot_o):,}) -- PROVISIONAL until step 08")
    o = pd.to_numeric(out["tokens_orig"], errors="coerce")
    if o.notna().sum() >= 8:
        q = pd.qcut(o, 4, labels=["smallest", "small", "large", "largest"], duplicates="drop")
        print(f"\n{'page-size bucket':<20}{'pages':>7}{'median tokens':>15}{'corpus red.':>13}")
        for b in q.cat.categories:
            sub = out[q == b]
            so = pd.to_numeric(sub["tokens_orig"], errors="coerce").sum()
            sf = pd.to_numeric(sub["tokens_final"], errors="coerce").sum()
            print(f"{str(b):<20}{len(sub):>7}{pd.to_numeric(sub['tokens_orig']).median():>15.0f}"
                  f"{100.0 * (so - sf) / max(so, 1):>12.2f}%")
    print(f"\n[07] wrote {sel_csv}\n[07] NEXT: python src/pipeline/08_final_validate.py --source {src}")


if __name__ == "__main__":
    main()