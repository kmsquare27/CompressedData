"""Step 00 -- download a small pilot set (HTML + screenshot pairs) and write a
frozen manifest CSV. The manifest, not the folder, defines your dataset.

Usage (from the repo root, venv active):
    python src/pipeline/00_download_pilot_data.py --source webcode2m --n 100

Outputs:
    data/raw_<source>/<id>.html + <id>.png
    data/splits/pilot_<source>_manifest.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402
from tqdm import tqdm  # noqa: E402

HTML_KEYS = ("text", "html", "code")


def _load_stream(source: str):
    from datasets import load_dataset
    if source == "webcode2m":
        return load_dataset("xcodemind/webcode2m_purified",
                            split="train", streaming=True)
    raise ValueError(f"unknown source: {source}")


def _keep(source: str, ex: dict) -> bool:
    if source == "webcode2m" and str(ex.get("lang", "en")) != "en":
        return False
    html = next((ex[k] for k in HTML_KEYS if k in ex and ex[k]), None)
    return html is not None and 300 <= len(html) <= 400_000 and ex.get("image") is not None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["webcode2m"], required=True)
    ap.add_argument("--n", type=int, default=100)
    args = ap.parse_args()

    raw_dir = ROOT / "data" / f"raw_{args.source}"
    raw_dir.mkdir(parents=True, exist_ok=True)
    (ROOT / "data" / "splits").mkdir(parents=True, exist_ok=True)
    manifest_path = ROOT / "data" / "splits" / f"pilot_{args.source}_manifest.csv"

    stream = _load_stream(args.source)
    rows, seen = [], 0
    first_keys_printed = False

    with tqdm(total=args.n, desc=f"downloading {args.source}") as bar:
        for ex in stream:
            if not first_keys_printed:
                print(f"[00] dataset columns: {sorted(ex.keys())}")
                first_keys_printed = True
            if not _keep(args.source, ex):
                continue
            html = next(ex[k] for k in HTML_KEYS if k in ex and ex[k])
            pid = f"{args.source[:2]}{seen:05d}"
            hp = raw_dir / f"{pid}.html"
            pp = raw_dir / f"{pid}.png"
            try:
                hp.write_text(html, encoding="utf-8")
                ex["image"].save(pp)  # PIL image from `datasets`
            except Exception as e:  # noqa: BLE001
                print(f"[00] skip {pid}: {e}")
                continue
            rows.append({"page_id": pid, "source": args.source,
                         "html_path": str(hp), "png_path": str(pp),
                         "html_chars": len(html)})
            seen += 1
            bar.update(1)
            if seen >= args.n:
                break

    pd.DataFrame(rows).to_csv(manifest_path, index=False)
    print(f"[00] wrote {len(rows)} pages -> {manifest_path}")
    print("[00] NEXT: python src/pipeline/01_determinism_audit.py --source", args.source)


if __name__ == "__main__":
    main()
