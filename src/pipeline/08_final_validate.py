"""Step 08 -- final validation of the selected dataset against the ORIGINAL.

Closes two holes a careful reviewer would find in v2:

  1. Nothing re-checked the SELECTED artifact against the original render.
     Each level was gated against its own base (L2 v2 even against a
     BeautifulSoup round-trip of it), so a stacked L1+L2(+L3) artifact could
     spend each level's tolerance separately and exceed the budget against
     the truth. Here every selected artifact is rendered and put through the
     frozen gate against ONE render of the original. If it fails, the next
     rung of step 07's ladder is tried; the original is the last rung.
  2. The fine-tuning manifest used to carry the DATASET's screenshot, while
     equivalence was proven under the harness (placeholder images, blocked
     remote assets, 1280px). The final manifest points at the harness render
     of the original, which is what the proof is about. Whether the dataset
     screenshot matches the harness render is recorded per page as a
     diagnostic (same size?) so the paper can state it.

Usage:
    python src/pipeline/08_final_validate.py --source webcode2m

Output: outputs/renders_final/<source>/<id>.png        training screenshots
        reports/csv/final_validation_<source>.csv
        data/splits/compressed_<source>_manifest.csv    THE fine-tuning set
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402
from tqdm import tqdm  # noqa: E402

from src.compare.gate import (TokenCounter, evaluate_pair, load_config,  # noqa: E402
                              render_page)
from src.render.harness import RenderHarness  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="webcode2m")
    ap.add_argument("--strict", action="store_true",
                    help="require pixel identity instead of the frozen gate")
    args = ap.parse_args()
    src = args.source

    sel = pd.read_csv(ROOT / "reports" / "csv" / f"level_selection_{src}.csv")
    sel["page_id"] = sel["page_id"].astype(str)
    render_dir = ROOT / "outputs" / "renders_final" / src
    render_dir.mkdir(parents=True, exist_ok=True)
    work = ROOT / "outputs" / "renders_gate" / src / "final"
    work.mkdir(parents=True, exist_ok=True)
    cfg = load_config(ROOT / "config" / "gate_config.yaml")
    tk = TokenCounter.get()
    rows = []

    with RenderHarness() as h:
        for _, r in tqdm(sel.iterrows(), total=len(sel), desc="final validation"):
            pid = r["page_id"]
            ladder = json.loads(r["ladder"])
            orig_path = ladder[-1]["html_path"]
            row = {"page_id": pid, "selected_level": r["level"]}
            try:
                art_o = render_page(h, orig_path, render_dir / f"{pid}.png", tk)
            except Exception as e:  # noqa: BLE001
                rows.append({**row, "status": f"failed_original: {e}"}); continue
            row["tokens_orig"] = art_o.tokens
            # Diagnostic: does the dataset screenshot even have the harness's size?
            try:
                dp = str(r.get("dataset_png_path", ""))
                if dp and Path(dp).exists():
                    with Image.open(dp) as im:
                        row["dataset_png_size"] = f"{im.width}x{im.height}"
                row["harness_png_size"] = f"{art_o.img.shape[1]}x{art_o.img.shape[0]}"
                row["dataset_png_same_size"] = row.get("dataset_png_size") == row["harness_png_size"]
            except Exception:  # noqa: BLE001
                pass

            final, rung_tried, first_ok = None, 0, None
            for rung in ladder:
                if rung["level"] == "original":
                    final = {"level": "original", "html_path": orig_path,
                             "tokens": art_o.tokens, "accepted": True, "pixel_identical": True}
                    break
                rung_tried += 1
                try:
                    art_c = render_page(h, rung["html_path"], work / f"{pid}_{rung['level']}.png", tk)
                    R = evaluate_pair(art_o, art_c, cfg, page_id=pid)
                except Exception as e:  # noqa: BLE001
                    row[f"{rung['level']}_error"] = str(e); continue
                ok = bool(R["accepted"]) and (bool(R["pixel_identical"]) or not args.strict)
                row[f"{rung['level']}_final_ok"] = ok
                row[f"{rung['level']}_pixel_identical"] = bool(R["pixel_identical"])
                if first_ok is None:
                    first_ok = ok
                if ok:
                    final = {"level": rung["level"], "html_path": rung["html_path"],
                             "tokens": art_c.tokens, "accepted": True,
                             "pixel_identical": bool(R["pixel_identical"]),
                             **{k: R[k] for k in ("height_delta_px", "text_ratio", "iou_mean",
                                                  "center_shift_max", "deltae_max") if k in R}}
                    break
            rows.append({**row, "status": "ok",
                         "first_rung_passed": bool(first_ok) if first_ok is not None else None,
                         "rungs_tried": rung_tried,
                         "final_level": final["level"], "html_path": final["html_path"],
                         "png_path": str(render_dir / f"{pid}.png"),
                         "tokens_final": final["tokens"],
                         "reduction_pct": round(100.0 * (art_o.tokens - final["tokens"])
                                                / max(art_o.tokens, 1), 3),
                         "final_pixel_identical": final.get("pixel_identical"),
                         **{k: v for k, v in final.items()
                            if k in ("height_delta_px", "text_ratio", "iou_mean",
                                     "center_shift_max", "deltae_max")}})

    out = pd.DataFrame(rows)
    rep = ROOT / "reports" / "csv"
    out.to_csv(rep / f"final_validation_{src}.csv", index=False)
    ok = out[out["status"] == "ok"]
    man = ok[["page_id", "png_path", "html_path", "final_level", "tokens_orig",
              "tokens_final", "reduction_pct"]].rename(columns={"final_level": "level"})
    man.insert(1, "source", src)
    man_csv = ROOT / "data" / "splits" / f"compressed_{src}_manifest.csv"
    man.to_csv(man_csv, index=False)

    print(f"\n[08] ---- final validation against the original render: {src} ----")
    if len(ok):
        fr = ok["first_rung_passed"].dropna()
        print(f"selected artifact passed as-is : {int(fr.astype(bool).sum())}/{len(fr)}  "
              f"<- the paper's 'every selected target re-validated' number")
        print(f"fell back one or more rungs    : {int((ok['rungs_tried'] > 1).sum())}")
        print(f"pixel-identical to original    : "
              f"{int(ok['final_pixel_identical'].astype(bool).sum())}/{len(ok)}")
        print(f"final level                    : "
              f"{ok['final_level'].value_counts().to_dict()}")
        to, tf = ok["tokens_orig"].sum(), ok["tokens_final"].sum()
        print(f"CORPUS-LEVEL token saving      : {100.0 * (to - tf) / max(to, 1):.2f}%  "
              f"({int(to - tf):,} of {int(to):,}) -- validated")
        if "dataset_png_same_size" in ok:
            s = ok["dataset_png_same_size"].dropna()
            print(f"dataset screenshot same size as harness render: "
                  f"{int(s.astype(bool).sum())}/{len(s)}  (training uses the harness render)")
    print(f"[08] tokenizer: {tk.name}")
    print(f"[08] wrote {man_csv}")


if __name__ == "__main__":
    main()
