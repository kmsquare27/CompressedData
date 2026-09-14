"""Export Step 08 artifacts into a portable paired bundle. Standard library only; never re-render.

Reconciles the final-validation report against the compressed manifest instead of
demanding that they contain identical ID sets, and records every excluded page with
its reason. Use --all to export the entire eligible cohort, or --plan-only to audit
counts before spending disk on a copy.
"""
from __future__ import annotations
import argparse
import datetime as dt
import json
import math
import random
import shutil
import time
from pathlib import Path
from sft_core import *

def resolve_source(root, value):
    if not nonempty(value):
        raise ValueError("Empty source path")
    p = Path(value)
    return p if p.is_absolute() else root / p

def read_id_file(path):
    return [s.strip() for s in Path(path).read_text(encoding="utf-8-sig").splitlines() if s.strip()]

def original_from_ladder(root, row):
    """Accept a ladder that repeats the original entry, provided every copy agrees."""
    ladder = json.loads(row["ladder"])
    paths = {str(resolve_source(root, x["html_path"]).resolve()) for x in ladder if x.get("level") == "original"}
    if len(paths) != 1:
        raise ValueError(f"selection ladder must name exactly one original HTML path, found {len(paths)}")
    return Path(paths.pop())

def export(args):
    # getattr defaults keep the historical caller/test signature working.
    for name, default in [("all", False), ("min_pages", None), ("plan_only", False),
                          ("ids", None), ("exclude_ids", None), ("zip", False), ("n", None),
                          ("acceptance_policy", "pixel-identical")]:
        if not hasattr(args, name):
            setattr(args, name, default)
    if args.acceptance_policy not in {"pixel-identical", "validated"}:
        raise ValueError("Unknown acceptance policy")
    if args.n is not None and (type(args.n) is not int or args.n < 1):
        raise ValueError("--n must be a positive integer")
    if args.min_pages is not None and (
        type(args.min_pages) is not int or args.min_pages < 1
    ):
        raise ValueError("--min-pages must be a positive integer")
    if args.all and (args.n is not None or args.ids):
        raise ValueError("--all cannot combine with --n or --ids")
    started = time.perf_counter()
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
    if not env_path.is_file():
        errors.append(f"Missing Step 08 environment: {env_path}")

    ok = {pid for pid, r in final.items() if r.get("status") == "ok"}
    orphan = sorted(set(man) - set(final))
    if orphan:
        # A compressed target with no validation record at all is a real inconsistency.
        errors.append(f"{len(orphan)} manifest page(s) have no final-validation row: {orphan[:10]}")
    missing_manifest = sorted(ok - set(man))
    if missing_manifest:
        errors.append(
            f"{len(missing_manifest)} validated page(s) have no compressed-manifest "
            f"row: {missing_manifest[:10]}"
        )

    def policy_reason(pid):
        r, m = final.get(pid), man.get(pid)
        if r is None:
            return "no_final_validation_row"
        if r.get("status") != "ok":
            return f"final_validation_status={r.get('status')}"
        if m is None:
            return "validated_but_absent_from_compressed_manifest"
        repeat = r.get("original_repeat_identical")
        if nonempty(repeat) and not truth(repeat):
            return "original_repeatability_failed"
        if (
            args.acceptance_policy == "pixel-identical"
            and not truth(r.get("final_pixel_identical"))
        ):
            return "not_pixel_identical_under_strict_export_policy"
        return None

    eligible = {
        pid for pid in ok & set(man) if policy_reason(pid) is None
    }

    excluded_ids = set()
    if getattr(args, "exclude_ids", None):
        excluded_ids = set(read_id_file(args.exclude_ids))
    requested = None
    if args.ids:
        requested = read_id_file(args.ids)
        if len(requested) != len(set(requested)):
            errors.append("Duplicate IDs in --ids file")
        if args.n is not None and len(requested) != args.n:
            errors.append(f"--ids contains {len(requested)} IDs, expected --n {args.n}")
        if set(requested) & excluded_ids:
            errors.append(f"Requested training IDs overlap reserved IDs: {sorted(set(requested)&excluded_ids)[:10]}")
        unusable = sorted(set(requested) - eligible)
        if unusable:
            errors.append(f"{len(unusable)} requested ID(s) are not finalized/eligible: {unusable[:10]}")
        ids = requested
    else:
        candidates = sorted(eligible - excluded_ids)
        if args.all:
            ids = candidates
        else:
            if args.n is None:
                errors.append("Pass --all, --ids or --n; there is no implied cohort size")
                ids = []
            elif len(candidates) < args.n:
                errors.append(f"Only {len(candidates)} eligible pages; requested {args.n}. Use --all, lower --n, or finalize more pages.")
                ids = []
            else:
                ids = sorted(random.Random(args.seed).sample(candidates, args.n))
    selected = set(ids)
    if not selected:
        errors.append("No pages selected; an empty bundle is not allowed")
    if args.min_pages is not None and len(selected) < args.min_pages:
        errors.append(f"Selected {len(selected)} pages, below --min-pages {args.min_pages}")

    # Reconciliation covers every ID seen anywhere, so nothing disappears silently.
    reconciliation, fallbacks = [], 0
    for pid in sorted(set(final) | set(man)):
        r, m = final.get(pid), man.get(pid)
        status = (r or {}).get("status")
        pixel = truth((r or {}).get("final_pixel_identical")) if r else None
        repeat = (r or {}).get("original_repeat_identical")
        fallback = bool(r and nonempty(r.get("original_sha256")) and r.get("original_sha256") == r.get("final_sha256"))
        rejection = policy_reason(pid)
        if rejection is not None:
            reason = rejection
        elif pid in selected:
            reason = "selected"
        elif pid in excluded_ids:
            reason = "reserved_by_exclude_ids"
        else:
            reason = "eligible_not_sampled"
        fallbacks += fallback and pid in selected
        reconciliation.append({"page_id": pid, "in_final_validation": r is not None,
                               "final_validation_status": status, "final_pixel_identical": pixel,
                               "original_repeat_identical": repeat,
                               "in_compressed_manifest": m is not None,
                               "in_level_selection": pid in sel, "in_pilot_manifest": pid in orig,
                               "is_original_fallback": fallback,
                               "final_level": (r or m or {}).get("final_level") or (m or {}).get("level"),
                               "tokens_orig": (r or {}).get("tokens_orig"),
                               "tokens_final": (r or {}).get("tokens_final"),
                               "eligible": pid in eligible, "selected": pid in selected, "reason": reason})

    counts = {"acceptance_policy": args.acceptance_policy,
              "final_validation_rows": len(final), "final_validation_ok": len(ok),
              "compressed_manifest_rows": len(man), "level_selection_rows": len(sel),
              "eligible_pages": len(eligible), "reserved_excluded": len(excluded_ids & eligible),
              "selected_pages": len(selected), "selected_original_fallbacks": int(fallbacks),
              "manifest_without_validation_row": len(orphan),
              "validated_without_manifest_row": len(ok - set(man)),
              "manifest_rows_not_validated": len(set(man) - ok)}
    print("Reconciliation:", json.dumps(counts, indent=2), flush=True)

    planned = []
    for position, pid in enumerate(ids):
        try:
            r, m = final[pid], man[pid]
            rejection = policy_reason(pid)
            if rejection is not None:
                raise ValueError(rejection)
            if nonempty(r.get("original_repeat_identical")) and not truth(r["original_repeat_identical"]):
                raise ValueError("original repeatability check failed")
            for a, b in [("original_sha256", "original_sha256"), ("final_sha256", "final_sha256"),
                         ("final_level", "level"), ("tokens_orig", "tokens_orig"), ("tokens_final", "tokens_final")]:
                if a.startswith("tokens_"):
                    left, right = float(r[a]), float(m[b])
                    same = (math.isfinite(left) and left.is_integer()
                            and left >= 0 and left == right)
                else:
                    same = nonempty(r.get(a)) and r[a] == m.get(b)
                if not same:
                    errors.append(f"{pid}: {a} mismatch between report and manifest")
            for k in ["html_path", "png_path"]:
                if resolve_source(root, r[k]).resolve() != resolve_source(root, m[k]).resolve():
                    errors.append(f"{pid}: {k} mismatch between report and manifest")
            if pid in sel:
                original = original_from_ladder(root, sel[pid])
            elif pid in orig:
                original = resolve_source(root, orig[pid]["html_path"])
            else:
                raise ValueError("original path missing from selection ladder and pilot manifest")
            paths = {"original": original, "verified": resolve_source(root, m["html_path"]),
                     "image": resolve_source(root, m["png_path"])}
            hashes = {}
            for key, p in paths.items():
                if not p.is_file():
                    raise ValueError(f"missing {p}; export where these paths exist")
                hashes[key+"_sha256"] = file_sha(p)
                if key != "image":
                    p.read_bytes().decode("utf-8", errors="strict")
                    expected = r["original_sha256" if key == "original" else "final_sha256"]
                    if hashes[key+"_sha256"] != expected:
                        errors.append(f"{pid}: {key} bytes changed since Step 08")
            planned.append((pid, paths, hashes, r))
        except Exception as e:
            errors.append(f"{pid}: {e}")
        if (position+1) % 250 == 0:
            print(f"Validated {position+1}/{len(ids)} pages ({time.perf_counter()-started:.0f}s)", flush=True)
    out.mkdir(parents=True)
    write_csv(out / "reconciliation.csv", reconciliation)
    write_csv(out / "excluded_pages.csv", [row for row in reconciliation if not row["selected"]])
    write_json(out / "reconciliation_summary.json", counts)
    write_json(out / "export_issues.json", {
        "validation_passed": not errors,
        "errors": errors,
        "acceptance_policy": args.acceptance_policy,
    })
    (out / "page_ids.txt").write_text(
        "\n".join(ids) + ("\n" if ids else ""), encoding="utf-8"
    )
    if errors:
        shown = errors[:40]
        more = f"\n- ... and {len(errors)-len(shown)} further error(s)" if len(errors) > len(shown) else ""
        raise ValueError(f"Export validation failed; no READY bundle written. Audit: {out}\n- " + "\n- ".join(shown) + more)

    if args.plan_only:
        print(f"PLAN ONLY: {len(ids)} eligible/selected pages validated; no page files copied. See {out}/reconciliation.csv")
        return
    rows = []
    for i, (pid, paths, hashes, r) in enumerate(planned):
        row = {"page_id": pid, "final_level": r["final_level"], **hashes,
               "is_original_fallback": hashes["original_sha256"] == hashes["verified_sha256"],
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
        if (i+1) % 250 == 0:
            print(f"Copied {i+1}/{len(planned)} pages ({time.perf_counter()-started:.0f}s)", flush=True)
    snapshots = out / "provenance"
    snapshots.mkdir()
    for p in [report, manifest, selection, pilot, env_path]:
        if p.is_file():
            shutil.copyfile(p, snapshots / p.name)
    write_jsonl(out / "pages.jsonl", rows)
    write_json(out / "bundle.json", {"schema": 1, "pages": len(rows), "seed": args.seed,
               "source": args.source, "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
               "pages_sha256": file_sha(out / "pages.jsonl"), "step08_environment": read_json(env_path),
               "selection_mode": "ids" if requested is not None else ("all" if args.all else "sample"),
               "acceptance_policy": args.acceptance_policy,
               "reconciliation": counts,
               "excluded_ids_sha256": file_sha(args.exclude_ids) if getattr(args, "exclude_ids", None) else None,
               "screenshot_provenance": "Step 08 original harness render; screenshot hash first recorded at export",
               "export_script_sha256": file_sha(__file__), "export_seconds": time.perf_counter()-started,
               "ready": True})
    if args.zip:
        if len(rows) > 500:
            print("Note: zipping a large bundle duplicates it on disk; skip --zip when the target pod shares this volume.", flush=True)
        archive = Path(str(out) + ".zip")
        if archive.exists():
            raise ValueError(f"Bundle saved, but archive exists: {archive}; not overwriting")
        shutil.make_archive(str(out), "zip", out.parent, out.name)
        print(f"Transfer: {archive}")
    print(f"Exported {len(rows)} paired pages ({counts['selected_original_fallbacks']} original fallbacks retained): {out}")

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", required=True, help="Compression project root")
    p.add_argument("--source", default="webcode2m")
    p.add_argument("--acceptance-policy",
                   choices=["pixel-identical", "validated"],
                   default="pixel-identical",
                   help="Strict pixel identity by default. validated accepts "
                        "Step 08 status=ok without adding a pixel-identity rule; "
                        "neither mode rerenders or changes the frozen gate.")
    p.add_argument("--n", type=int, default=None, help="Sample exactly N eligible pages; omit when using --all or --ids")
    p.add_argument("--all", action="store_true", help="Export every eligible page not reserved by --exclude-ids")
    p.add_argument("--min-pages", type=int, default=None, help="Fail if fewer pages are selected than this")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--ids", help="Optional exact TRAIN page IDs, one per line")
    p.add_argument("--exclude-ids", help="Reserved validation/test IDs, one per line; never train these")
    p.add_argument("--plan-only", action="store_true", help="Validate and write reconciliation only; copy no page files")
    p.add_argument("--out", required=True)
    p.add_argument("--zip", action="store_true")
    a = p.parse_args()
    if a.n is not None and a.n < 1:
        p.error("--n must be positive")
    if a.all and a.n is not None:
        p.error("Use either --all or --n")
    if a.all and a.ids:
        p.error("Use either --all or --ids")
    export(a)

if __name__ == "__main__":
    main()
