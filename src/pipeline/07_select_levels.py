"""Step 07 -- per-page level selection and the final compressed dataset.

Every page keeps a training target. Rejection never drops a page, it only
means that page's target is less compressed:

    L1 accepted, L2 accepted   -> outputs/level2/<id>.html  (L1+L2 stacked)
    L1 accepted, L2 rejected   -> outputs/level1/<id>.html  (L1 only)
    L1 rejected, L2 accepted   -> outputs/level2/<id>.html  (L2 only, built
                                  from the ORIGINAL because step 06 refuses
                                  to compose on a rejected L1)
    both rejected              -> the original HTML, 0% reduction

SELECTION IS ON ABSOLUTE TOKENS, NOT reduction_pct. The two gate CSVs measure
percentages against different baselines -- L2's is relative to whatever file
it started from -- so comparing them directly would silently prefer the wrong
artifact whenever L2 ran on the original. Absolute token counts are all
against the same page, so min() is well-defined.

The comparison is safe by construction: every candidate here already passed
the frozen gate, G1 included, so each is both shorter than its own input and
render-validated. Taking the fewest-token accepted artifact per page
therefore yields the largest corpus-level saving that render validation
permits.

Usage:
    python src/pipeline/07_select_levels.py --source webcode2m

Output: data/splits/compressed_<source>_manifest.csv   the fine-tuning set
        reports/csv/level_selection_<source>.csv       full audit detail
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402


def _accepted(df: pd.DataFrame) -> pd.Series:
    if "accepted" not in df.columns:
        return pd.Series(False, index=df.index)
    return df["accepted"].astype(str).str.lower().isin(["true", "1"])


def _num(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(float("nan"), index=df.index)
    return pd.to_numeric(df[col], errors="coerce")


def load_level(path: Path, level: str) -> pd.DataFrame:
    """One row per page: was it accepted, and how many tokens did it end at."""
    if not path.exists():
        print(f"[07] no {path.name}; treating {level} as unavailable")
        return pd.DataFrame(columns=["page_id", f"{level}_ok", f"{level}_tokens",
                                     f"{level}_base"])
    d = pd.read_csv(path)
    out = pd.DataFrame({
        "page_id": d["page_id"].astype(str),
        f"{level}_ok": _accepted(d),
        f"{level}_tokens": _num(d, "tokens_comp"),
        f"{level}_orig_tokens": _num(d, "tokens_orig"),
    })
    out[f"{level}_base"] = (d["base"].astype(str) if "base" in d.columns
                            else "original")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="webcode2m")
    args = ap.parse_args()
    src = args.source

    man = ROOT / "data" / "splits" / f"pilot_{src}_manifest.csv"
    if not man.exists():
        print(f"[07] no manifest at {man}"); return
    m = pd.read_csv(man)
    m["page_id"] = m["page_id"].astype(str)

    rep = ROOT / "reports" / "csv"
    d1 = load_level(rep / f"level1_gate_{src}.csv", "l1")
    d2 = load_level(rep / f"level2_gate_{src}.csv", "l2")
    df = m.merge(d1, on="page_id", how="left").merge(d2, on="page_id", how="left")

    # Original token count: L1 always measures it against the true original.
    # L2 only does when it ran on the original (base == "original").
    orig = df["l1_orig_tokens"]
    from_l2 = df["l2_orig_tokens"].where(df["l2_base"] == "original")
    orig = orig.fillna(from_l2)

    l1_dir = ROOT / "outputs" / "level1" / src
    l2_dir = ROOT / "outputs" / "level2" / src

    rows = []
    for _, r in df.iterrows():
        pid = r["page_id"]
        o = r.get("html_path")
        t_orig = orig.loc[r.name]
        cands = [("original", o, t_orig)]
        if bool(r.get("l1_ok")) and (l1_dir / f"{pid}.html").exists():
            cands.append(("l1", str(l1_dir / f"{pid}.html"), r["l1_tokens"]))
        if bool(r.get("l2_ok")) and (l2_dir / f"{pid}.html").exists():
            cands.append(("l2", str(l2_dir / f"{pid}.html"), r["l2_tokens"]))

        usable = [c for c in cands if pd.notna(c[2])]
        if not usable:                       # no token counts at all
            level, path, tok = "original", o, float("nan")
        else:
            level, path, tok = min(usable, key=lambda c: c[2])

        red = (100.0 * (t_orig - tok) / t_orig
               if pd.notna(t_orig) and pd.notna(tok) and t_orig else 0.0)
        rows.append({
            "page_id": pid, "source": src, "png_path": r.get("png_path", ""),
            "html_path": path, "level": level,
            "tokens_orig": t_orig, "tokens_final": tok,
            "reduction_pct": round(float(red), 3),
            "l1_ok": bool(r.get("l1_ok")), "l2_ok": bool(r.get("l2_ok")),
            "l2_base": r.get("l2_base", ""),
        })

    out = pd.DataFrame(rows)
    sel_csv = rep / f"level_selection_{src}.csv"
    out.to_csv(sel_csv, index=False)
    man_csv = ROOT / "data" / "splits" / f"compressed_{src}_manifest.csv"
    out[["page_id", "source", "png_path", "html_path", "level",
         "tokens_orig", "tokens_final", "reduction_pct"]].to_csv(man_csv,
                                                                 index=False)

    # ---- report -----------------------------------------------------------
    n = len(out)
    print(f"\n[07] ---- final dataset: {src} ({n} pages, none dropped) ----\n")
    print(f"{'case':<34}{'pages':>7}{'mean red.':>11}")
    cases = [
        ("L1 + L2 stacked", (out.l1_ok) & (out.l2_ok) & (out.level == "l2")),
        ("L1 only", (out.level == "l1")),
        ("L2 only (L1 had failed)", (~out.l1_ok) & (out.level == "l2")),
        ("no accepted compression", (out.level == "original")),
    ]
    for name, mask in cases:
        sub = out[mask]
        red = sub["reduction_pct"].mean() if len(sub) else 0.0
        print(f"{name:<34}{len(sub):>7}{red:>10.2f}%")

    comp = out[out.level != "original"]
    tot_o = pd.to_numeric(out["tokens_orig"], errors="coerce").sum()
    tot_f = pd.to_numeric(out["tokens_final"], errors="coerce").sum()
    print(f"\ncompressed pages           : {len(comp)}/{n} "
          f"({100.0 * len(comp) / max(n, 1):.0f}%)")
    if len(comp):
        print(f"reduction on those pages   : mean "
              f"{comp['reduction_pct'].mean():.2f}%  "
              f"median {comp['reduction_pct'].median():.2f}%  "
              f"p90 {comp['reduction_pct'].quantile(.9):.2f}%")
    if tot_o:
        print(f"CORPUS-LEVEL token saving  : {100.0 * (tot_o - tot_f) / tot_o:.2f}%"
              f"   ({int(tot_o - tot_f):,} of {int(tot_o):,} tokens)")
        print("  ^ the honest headline: every page counted, rejections at 0%.")
        print("    Quote the per-page figure only WITH the acceptance rate, or "
              "it reads as conditioning on the outcome.")

    # Size buckets: where the redundancy actually lives.
    o = pd.to_numeric(out["tokens_orig"], errors="coerce")
    if o.notna().sum() >= 8:
        q = pd.qcut(o, 4, labels=["smallest", "small", "large", "largest"],
                    duplicates="drop")
        print(f"\n{'page-size bucket':<20}{'pages':>7}{'median tokens':>15}"
              f"{'corpus red.':>13}")
        for b in q.cat.categories:
            sub = out[q == b]
            so = pd.to_numeric(sub["tokens_orig"], errors="coerce").sum()
            sf = pd.to_numeric(sub["tokens_final"], errors="coerce").sum()
            print(f"{str(b):<20}{len(sub):>7}"
                  f"{pd.to_numeric(sub['tokens_orig']).median():>15.0f}"
                  f"{100.0 * (so - sf) / max(so, 1):>12.2f}%")

    bad = out[pd.to_numeric(out["tokens_final"], errors="coerce")
              > pd.to_numeric(out["tokens_orig"], errors="coerce")]
    if len(bad):
        print(f"\n[07] WARNING: {len(bad)} selected targets are LONGER than "
              f"the original -- G1 should make this impossible; inspect "
              f"{sel_csv.name}")

    print(f"\n[07] wrote {man_csv}")
    print(f"[07] wrote {sel_csv}")


if __name__ == "__main__":
    main()