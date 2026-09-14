"""Prepare selected paired arms (default: original, verified); never truncate."""
from __future__ import annotations
import argparse
import datetime as dt
import math
import shutil
import time
from pathlib import Path
from sft_core import *

def percentile(values, q):
    """Nearest-rank percentile; no numpy dependency in the preparation step."""
    ordered = sorted(values)
    return ordered[min(len(ordered)-1, max(0, math.ceil(q/100*len(ordered))-1))]

MINIFY_OPTIONS = {"minify_css": True, "minify_js": False, "keep_closing_tags": True,
                  "keep_html_and_head_opening_tags": True, "minify_doctype": False}

def prepare(args):
    for name in ("expect_pages", "max_overflow_pages"):
        if not hasattr(args, name):
            setattr(args, name, None)
    if args.expect_pages is not None and (
        type(args.expect_pages) is not int or args.expect_pages < 1
    ):
        raise ValueError("--expect-pages must be a positive integer")
    if args.max_overflow_pages is not None and (
        type(args.max_overflow_pages) is not int or args.max_overflow_pages < 0
    ):
        raise ValueError("--max-overflow-pages must be a nonnegative integer")
    from PIL import Image
    requested = list(getattr(args, "arms", ["original", "verified"]))
    if (len(requested) < 2 or len(set(requested)) != len(requested)
            or "original" not in requested or not set(requested) <= set(ARMS)):
        raise ValueError("Choose original and at least one other supported arm, without duplicates")
    # Canonical order makes the prepared identity independent of CLI arm order.
    arms = [arm for arm in ARMS if arm in requested]
    started = time.perf_counter()
    out, bundle = Path(args.out).resolve(), Path(args.bundle).resolve()
    if out.exists():
        raise ValueError("Prepared output exists; use a new directory")
    c, lock = load_config(args.config), read_json(args.model_lock)
    model_dir = verify_model_lock(lock)
    meta = read_json(bundle / "bundle.json")
    if not meta.get("ready") or file_sha(bundle / "pages.jsonl") != meta["pages_sha256"]:
        raise ValueError("Bundle incomplete or pages manifest changed")
    rows = read_jsonl(bundle / "pages.jsonl")
    errors = validate_bundle_rows(bundle, rows)
    if not rows or meta["pages"] != len(rows):
        errors.append("Empty bundle or page count mismatch")
    if errors:
        raise ValueError("Bundle validation failed:\n- " + "\n- ".join(errors))
    processor = load_processor(model_dir, c)
    if "naive" in arms:
        import minify_html
        # Probe only for an explicitly requested naive arm.
        minify_html.minify("<!doctype html><html><head></head><body>probe</body></html>", **MINIFY_OPTIONS)
    image_id = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
    out.mkdir(parents=True)
    records, coverage, errors = [], [], []
    excluded = set()
    for i, row in enumerate(rows):
        pid = row["page_id"]
        try:
            dest = out / f"pages/{i:06d}"
            dest.mkdir(parents=True)
            for key in ["image", "original", "verified"]:
                copied = dest / (key + (".png" if key == "image" else ".html"))
                shutil.copyfile(portable_path(bundle, row[key]), copied)
                if file_sha(copied) != row[key + "_sha256"]:
                    raise ValueError(f"Copied {key} bytes differ from the frozen bundle")
            naive_error = None
            if "naive" in arms:
                # Preserve original bytes; match Step 08's universal-newline text loading.
                original = (dest / "original.html").read_text(encoding="utf-8")
                try:
                    naive = minify_html.minify(original, **MINIFY_OPTIONS)
                except Exception as e:
                    # Predetermined exception policy, never a visual-quality decision.
                    naive, naive_error = original, repr(e)
                (dest / "naive.html").write_text(naive, encoding="utf-8", newline="\n")
            with Image.open(dest / "image.png") as opened:
                im = opened.convert("RGB")
            try:
                prefix_text = conversation_text(processor)
                prefix = encode_image_text(processor, im, prefix_text)
                prefix_ids = prefix["input_ids"][0].tolist()
                for arm in arms:
                    html_path = dest / f"{arm}.html"
                    html = html_path.read_text(encoding="utf-8")
                    if not html.strip():
                        errors.append(f"{pid}/{arm}: empty HTML target")
                        continue
                    forbidden = [s for s in processor.tokenizer.all_special_tokens if s and s in html]
                    if forbidden:
                        errors.append(f"{pid}/{arm}: target contains reserved model token literals {forbidden}")
                        continue
                    full_text = conversation_text(processor, html)
                    enc = encode_image_text(processor, im, full_text)
                    ids = enc["input_ids"][0].tolist()
                    labels, boundary_whitespace = assistant_labels(processor, ids, prefix_ids, full_text, prefix_text, image_id)
                    prefix_tokens = next(j for j,x in enumerate(labels) if x != -100)
                    count = sum(x != -100 for x in labels[1:])
                    image_tokens = ids.count(image_id)
                    grid = enc["image_grid_thw"].tolist()
                    rec = {"page_id": pid, "arm": arm,
                           "image": str((dest / "image.png").relative_to(out)),
                           "html": str(html_path.relative_to(out)),
                           "html_sha256": file_sha(html_path), "image_sha256": row["image_sha256"],
                           "input_ids_sha256": digest(ids), "prefix_tokens": prefix_tokens,
                           "masked_boundary_whitespace_chars": boundary_whitespace,
                           "sequence_tokens": len(ids), "image_tokens": image_tokens,
                           "prompt_nonimage_tokens": prefix_tokens-image_tokens,
                           "target_tokens": count, "padding_tokens": 0,
                           "html_standalone_tokens": len(processor.tokenizer.encode(html, add_special_tokens=False)),
                           "image_grid_thw": grid,
                           "vision_patch_tokens": sum(t*h*w for t,h,w in grid),
                           "final_level": row["final_level"], "naive_exception": naive_error if arm == "naive" else None}
                    if rec["prompt_nonimage_tokens"] + image_tokens + count != len(ids):
                        raise ValueError("Token accounting partition failed")
                    overflow = len(ids) > c["max_seq_length"]
                    if overflow:
                        excluded.add(pid)
                    coverage.append({k: rec[k] for k in ["page_id", "arm", "sequence_tokens", "image_tokens", "prompt_nonimage_tokens", "target_tokens", "html_standalone_tokens", "vision_patch_tokens"]} | {"over_limit": overflow})
                    records.append(rec)
                    del enc
            finally:
                im.close()
            if (i+1) % 25 == 0 or (i+1) == len(rows):
                spent = time.perf_counter()-started
                left = spent/(i+1)*(len(rows)-i-1)
                print(f"Prepared {i+1}/{len(rows)} pages ({spent/60:.1f} min elapsed, ~{left/60:.1f} min left)", flush=True)
        except Exception as e:
            errors.append(f"{pid}: {e}")
    write_csv(out / "coverage.csv", coverage)
    overflow_detail = sorted(({k: row[k] for k in ["page_id", "arm", "sequence_tokens", "image_tokens", "target_tokens"]}
                              for row in coverage if row["page_id"] in excluded),
                             key=lambda row: (row["page_id"], row["arm"]))
    retained_count = len({
        row["page_id"] for row in records
        if row["arm"] == "original" and row["page_id"] not in excluded
    })
    if args.expect_pages is not None and retained_count != args.expect_pages:
        errors.append(
            f"Expected {args.expect_pages} retained paired pages; "
            f"preflight found {retained_count}"
        )
    write_json(out / "preparation_issues.json", {
                                                "expect_pages": args.expect_pages,
                                                "candidate_retained_pages": retained_count,
                                                "errors": errors, "overflow_page_ids": sorted(excluded),
                                                "overflow_detail": overflow_detail,
                                                "max_seq_length": c["max_seq_length"],
                                                "overflow_policy": args.overflow_policy,
                                                "max_overflow_pages": args.max_overflow_pages, "arms": arms})
    if errors or (excluded and args.overflow_policy == "error"):
        raise ValueError(f"Preflight failed: {len(errors)} error(s), {len(excluded)} page(s) over limit. See {out}/preparation_issues.json and coverage.csv. No READY dataset written.")
    if args.max_overflow_pages is not None and len(excluded) > args.max_overflow_pages:
        raise ValueError(f"{len(excluded)} page(s) exceed max_seq_length={c['max_seq_length']}, above the declared budget of {args.max_overflow_pages}. Review {out}/preparation_issues.json before changing the policy. No READY dataset written.")
    records = [r for r in records if r["page_id"] not in excluded]
    if not records:
        raise ValueError("No common pages fit; adjust sequence/image policy and re-prepare")
    for arm in arms:
        part = [r for r in records if r["arm"] == arm]
        write_jsonl(out / f"{arm}.jsonl", part)
    ids = [r["page_id"] for r in records if r["arm"] == "original"]
    if not all(
        [r["page_id"] for r in records if r["arm"] == arm] == ids
        for arm in arms
    ):
        raise ValueError("Selected arm page cohorts differ; no READY dataset written")
    summary = []
    for arm in arms:
        subset = [r for r in records if r["arm"] == arm]
        summary.append({"arm": arm, "pages": len(subset), **{key: sum(r[key] for r in subset) for key in
                        ["sequence_tokens", "target_tokens", "image_tokens", "prompt_nonimage_tokens", "html_standalone_tokens", "vision_patch_tokens"]},
                        "max_sequence_tokens": max(r["sequence_tokens"] for r in subset),
                        **{f"p{q}_sequence_tokens": percentile([r["sequence_tokens"] for r in subset], q)
                           for q in (50, 90, 99)},
                        "headroom_to_cap_tokens": c["max_seq_length"]-max(r["sequence_tokens"] for r in subset),
                        "naive_exception_count": sum(r["naive_exception"] is not None for r in subset)})
    write_csv(out / "token_summary.csv", summary)
    processor.save_pretrained(out / "processor")
    contract = preprocessing_contract(c)
    metadata = {"schema": 1, "ready": True, "pages": len(ids), "page_ids": ids, "arms": arms,
                "input_pages": len(rows), "excluded_page_ids": sorted(excluded),
                "overflow_policy": args.overflow_policy, "preprocessing": contract,
                "expect_pages": args.expect_pages,
                "max_overflow_pages": args.max_overflow_pages,
                "model_id": lock["model_id"], "model_revision": lock["revision"],
                "minify_options": MINIFY_OPTIONS if "naive" in arms else None, "packages": package_versions(),
                "bundle": meta, "preparation_seconds": time.perf_counter()-started,
                "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                "manifest_hashes": {arm: file_sha(out / f"{arm}.jsonl") for arm in arms}}
    metadata["data_fingerprint"] = digest({"contract": contract, "revision": lock["revision"],
                                           "arms": arms, "manifests": metadata["manifest_hashes"]})
    write_json(out / "prepared.json", metadata)
    print(f"READY: {len(ids)} TRAIN pages per arm ({', '.join(arms)}); {out}")

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bundle", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--model-lock", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--arms", nargs="+", choices=ARMS, default=["original", "verified"],
                   help="Default: original verified. Include naive explicitly for a three-arm experiment.")
    p.add_argument("--overflow-policy", choices=["error", "common-fit"], default="error",
                   help="Default fails. common-fit removes a page from ALL SELECTED arms if any selected arm exceeds the cap.")
    p.add_argument("--max-overflow-pages", type=int, default=None,
                   help="With common-fit, fail instead of silently shrinking the cohort beyond this many excluded pages.")
    p.add_argument("--expect-pages", type=int, default=None,
                   help="Fail unless READY contains exactly this many paired pages.")
    prepare(p.parse_args())

if __name__ == "__main__":
    main()
