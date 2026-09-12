"""Export Step 08 artifacts on the source PC. Standard library only; never re-render."""
from __future__ import annotations
import argparse
import datetime as dt
import json
import random
import shutil
from pathlib import Path
from sft_core import *

def resolve_source(root, value):
    if not nonempty(value):
        raise ValueError("Empty source path")
    p = Path(value)
    return p if p.is_absolute() else root / p

def export(args):
    root, out = Path(args.root).resolve(), Path(args.out).resolve()
    if out.exists():
        raise ValueError(f"Output already exists: {out}; choose a new path")
    report = root / f"reports/csv/final_validation_{args.source}.csv"
    manifest = root / f"data/splits/compressed_{args.source}_manifest.csv"
    selection = root / f"reports/csv/level_selection_{args.source}.csv"
    pilot = root / f"data/splits/pilot_{args.source}_manifest.csv"
    env_path = root / f"reports/csv/final_validation_{args.source}_environment.json"
    errors = []
    final = unique_index(read_csv(report), "final validation", errors)
    man = unique_index(read_csv(manifest), "training manifest", errors)
    orig = unique_index(read_csv(pilot, required=False), "pilot", errors)
    sel = unique_index(read_csv(selection, required=False), "selection", errors)
    ok = {pid for pid, r in final.items() if r.get("status") == "ok"}
    if ok != set(man):
        errors.append(f"Step 08/manifest IDs differ: report-only {sorted(ok-set(man))[:10]}, manifest-only {sorted(set(man)-ok)[:10]}")
    if not env_path.is_file():
        errors.append(f"Missing Step 08 environment: {env_path}")
    excluded_ids = set()
    if getattr(args, "exclude_ids", None):
        excluded_ids = {s.strip() for s in Path(args.exclude_ids).read_text(encoding="utf-8-sig").splitlines() if s.strip()}
    if args.ids:
        ids = [s.strip() for s in Path(args.ids).read_text(encoding="utf-8-sig").splitlines() if s.strip()]
        if len(ids) != len(set(ids)):
            errors.append("Duplicate IDs in --ids file")
        if len(ids) != args.n:
            errors.append(f"--ids contains {len(ids)} IDs, expected --n {args.n}")
        if set(ids) & excluded_ids:
            errors.append(f"Requested training IDs overlap reserved IDs: {sorted(set(ids)&excluded_ids)[:10]}")
    else:
        candidates = sorted((ok & set(man))-excluded_ids)
        if len(candidates) < args.n:
            errors.append(f"Only {len(candidates)} finalized pages; requested {args.n}. Finalize more or explicitly change --n.")
        ids = random.Random(args.seed).sample(candidates, min(args.n, len(candidates)))
    planned = []
    for pid in ids:
        try:
            r, m = final[pid], man[pid]
            if r.get("status") != "ok" or not truth(r.get("final_pixel_identical")):
                raise ValueError("requires Step 08 status=ok and final_pixel_identical=True")
            if nonempty(r.get("original_repeat_identical")) and not truth(r["original_repeat_identical"]):
                raise ValueError("original repeatability check failed")
            for a, b in [("original_sha256", "original_sha256"), ("final_sha256", "final_sha256"),
                         ("final_level", "level"), ("tokens_orig", "tokens_orig"), ("tokens_final", "tokens_final")]:
                if a.startswith("tokens_"):
                    same = float(r[a]) == float(m[b]) and float(r[a]) >= 0
                else:
                    same = nonempty(r.get(a)) and r[a] == m.get(b)
                if not same:
                    errors.append(f"{pid}: {a} mismatch between report and manifest")
            for k in ["html_path", "png_path"]:
                if resolve_source(root, r[k]).resolve() != resolve_source(root, m[k]).resolve():
                    errors.append(f"{pid}: {k} mismatch between report and manifest")
            original = None
            if pid in sel:
                ladder = json.loads(sel[pid]["ladder"])
                originals = [x for x in ladder if x.get("level") == "original"]
                if len(originals) != 1:
                    raise ValueError("selection ladder must contain exactly one original")
                original = resolve_source(root, originals[0]["html_path"])
            elif pid in orig:
                original = resolve_source(root, orig[pid]["html_path"])
            else:
                raise ValueError("original path missing from selection ladder and pilot manifest")
            paths = {"original": original, "verified": resolve_source(root, m["html_path"]),
                     "image": resolve_source(root, m["png_path"])}
            hashes = {}
            for key, p in paths.items():
                if not p.is_file():
                    raise ValueError(f"missing {p}; export on the PC where these paths exist")
                hashes[key+"_sha256"] = file_sha(p)
                if key != "image":
                    p.read_bytes().decode("utf-8", errors="strict")
                    expected = r["original_sha256" if key == "original" else "final_sha256"]
                    if hashes[key+"_sha256"] != expected:
                        errors.append(f"{pid}: {key} bytes changed since Step 08")
            planned.append((pid, paths, hashes, r))
        except Exception as e:
            errors.append(f"{pid}: {e}")
    if errors:
        raise ValueError("Export validation failed; no bundle written:\n- " + "\n- ".join(errors))
    out.mkdir(parents=True)
    rows = []
    for i, (pid, paths, hashes, r) in enumerate(planned):
        row = {"page_id": pid, "final_level": r["final_level"], **hashes,
               "step08_tokens_orig": int(float(r["tokens_orig"])), "step08_tokens_final": int(float(r["tokens_final"]))}
        for key, p in paths.items():
            relative = f"pages/{i:06d}/{key}{'.png' if key == 'image' else '.html'}"
            dest = out / relative
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(p, dest)
            if file_sha(dest) != hashes[key+"_sha256"]:
                raise ValueError(f"Source changed during export: {pid}/{key}; discard incomplete bundle")
            row[key] = relative
        rows.append(row)
    snapshots = out / "provenance"
    snapshots.mkdir()
    for p in [report, manifest, selection, pilot, env_path]:
        if p.is_file():
            shutil.copyfile(p, snapshots / p.name)
    write_jsonl(out / "pages.jsonl", rows)
    write_json(out / "bundle.json", {"schema": 1, "pages": len(rows), "seed": args.seed,
               "source": args.source, "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
               "pages_sha256": file_sha(out / "pages.jsonl"), "step08_environment": read_json(env_path),
               "excluded_ids_sha256": file_sha(args.exclude_ids) if getattr(args,"exclude_ids",None) else None,
               "screenshot_provenance": "Step 08 original harness render; screenshot hash first recorded at export",
               "export_script_sha256": file_sha(__file__), "ready": True})
    if args.zip:
        archive = Path(str(out) + ".zip")
        if archive.exists():
            raise ValueError(f"Bundle saved, but archive exists: {archive}; not overwriting")
        shutil.make_archive(str(out), "zip", out.parent, out.name)
        print(f"Transfer: {archive}")
    print(f"Exported {len(rows)} paired pages: {out}")

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", required=True, help="Compression project root on source PC")
    p.add_argument("--source", default="webcode2m")
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--ids", help="Optional exact TRAIN page IDs, one per line")
    p.add_argument("--exclude-ids", help="Optional reserved validation/test IDs, one per line; never train these")
    p.add_argument("--out", required=True)
    p.add_argument("--zip", action="store_true")
    a = p.parse_args()
    if a.n < 1:
        p.error("--n must be positive")
    export(a)

if __name__ == "__main__":
    main()
