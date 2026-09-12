"""Build all three paired arms; measure processor-expanded tokens, never truncate."""
from __future__ import annotations
import argparse
import datetime as dt
import shutil
import time
from pathlib import Path
from sft_core import *

MINIFY_OPTIONS = {"minify_css": True, "minify_js": False, "keep_closing_tags": True,
                  "keep_html_and_head_opening_tags": True, "minify_doctype": False}

def prepare(args):
    from PIL import Image
    import minify_html
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
    # Catch an incompatible minifier API as an environment failure, not 100 silent fallbacks.
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
                shutil.copyfile(portable_path(bundle, row[key]), dest / (key + (".png" if key == "image" else ".html")))
            # Match Step 08 text loading's universal-newline behavior; preserve original file bytes on disk.
            original = (dest / "original.html").read_text(encoding="utf-8")
            naive_error = None
            try:
                naive = minify_html.minify(original, **MINIFY_OPTIONS)
            except Exception as e:
                # Predetermined exception policy, never a gate or visual-quality decision.
                naive, naive_error = original, repr(e)
            (dest / "naive.html").write_text(naive, encoding="utf-8", newline="\n")
            with Image.open(dest / "image.png") as opened:
                im = opened.convert("RGB")
            try:
                prefix_text = conversation_text(processor)
                prefix = encode_image_text(processor, im, prefix_text)
                prefix_ids = prefix["input_ids"][0].tolist()
                for arm in ARMS:
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
            if (i+1) % 10 == 0:
                print(f"Prepared {i+1}/{len(rows)} pages", flush=True)
        except Exception as e:
            errors.append(f"{pid}: {e}")
    write_csv(out / "coverage.csv", coverage)
    write_json(out / "preparation_issues.json", {"errors": errors, "overflow_page_ids": sorted(excluded),
                                                "overflow_policy": args.overflow_policy})
    if errors or (excluded and args.overflow_policy == "error"):
        raise ValueError(f"Preflight failed: {len(errors)} error(s), {len(excluded)} page(s) over limit. See {out}/preparation_issues.json and coverage.csv. No READY dataset written.")
    records = [r for r in records if r["page_id"] not in excluded]
    if not records:
        raise ValueError("No common pages fit; adjust sequence/image policy and re-prepare")
    for arm in ARMS:
        part = [r for r in records if r["arm"] == arm]
        write_jsonl(out / f"{arm}.jsonl", part)
    ids = [r["page_id"] for r in records if r["arm"] == "original"]
    assert all([r["page_id"] for r in records if r["arm"] == arm] == ids for arm in ARMS)
    summary = []
    for arm in ARMS:
        subset = [r for r in records if r["arm"] == arm]
        summary.append({"arm": arm, "pages": len(subset), **{key: sum(r[key] for r in subset) for key in
                        ["sequence_tokens", "target_tokens", "image_tokens", "prompt_nonimage_tokens", "html_standalone_tokens", "vision_patch_tokens"]},
                        "max_sequence_tokens": max(r["sequence_tokens"] for r in subset),
                        "naive_exception_count": sum(r["naive_exception"] is not None for r in subset)})
    write_csv(out / "token_summary.csv", summary)
    processor.save_pretrained(out / "processor")
    contract = preprocessing_contract(c)
    metadata = {"schema": 1, "ready": True, "pages": len(ids), "page_ids": ids,
                "input_pages": len(rows), "excluded_page_ids": sorted(excluded),
                "overflow_policy": args.overflow_policy, "preprocessing": contract,
                "model_id": lock["model_id"], "model_revision": lock["revision"],
                "minify_options": MINIFY_OPTIONS, "packages": package_versions(),
                "bundle": meta, "preparation_seconds": time.perf_counter()-started,
                "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                "manifest_hashes": {arm: file_sha(out / f"{arm}.jsonl") for arm in ARMS}}
    metadata["data_fingerprint"] = digest({"contract": contract, "revision": lock["revision"],
                                           "manifests": metadata["manifest_hashes"]})
    write_json(out / "prepared.json", metadata)
    print(f"READY: {len(ids)} TRAIN pages per arm; {out}")

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bundle", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--model-lock", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--overflow-policy", choices=["error", "common-fit"], default="error",
                   help="Default fails. common-fit explicitly removes a page from ALL arms if any arm exceeds the cap.")
    prepare(p.parse_args())

if __name__ == "__main__":
    main()
