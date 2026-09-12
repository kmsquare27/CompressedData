"""Generate held-out HTML and profile single-request GPU inference.

Install in qwen_sft_patch/compute_tools; imports the existing parent sft_core.
Reads a Step 12 HELD-OUT bundle directly (no target-length filtering).
Only image and fixed training prompt are passed to the model. One adapter/run
per process; run each arm with identical arguments and a fresh output folder.
"""
from __future__ import annotations
import argparse
import gc
import math
import random
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sft_core import (ARMS, append_jsonl, conversation_text, digest, encode_image_text,
                      file_sha, load_processor, package_versions,
                      portable_path, preprocessing_contract, read_json, read_jsonl,
                      validate_bundle_rows, verify_model_lock, write_csv, write_json)


class TokenClock:
    """Host token-arrival timestamps, not text chunks or isolated GPU kernels.

    Transformers sends the prompt first, then one CPU token per greedy step.
    Reject a changed callback contract instead of silently misreporting TTFT.
    """
    def __init__(self, prompt_ids, clock=time.perf_counter):
        self.prompt_ids = prompt_ids
        self.clock = clock
        self.prompt_seen = False
        self.ids = []
        self.times = []

    def put(self, value):
        values = value.detach().cpu().reshape(-1).tolist()
        if not self.prompt_seen:
            if values != self.prompt_ids:
                raise ValueError("Streamer prompt mismatch")
            self.prompt_seen = True
            return
        if len(values) != 1:
            raise ValueError("This profiler requires batch=1, beams=1, one token per callback")
        self.ids.extend(values)
        self.times.append(self.clock())

    def end(self):
        pass


def token_metrics(ids, times, started, finished, eos_ids, budget):
    if not ids or len(ids) != len(times) or finished < started:
        raise ValueError("Missing or inconsistent generation timing")
    if any(t < started or t > finished for t in times) or any(b < a for a,b in zip(times, times[1:])):
        raise ValueError("Non-monotonic token timing")
    decode = times[-1]-times[0]
    eos = ids[-1] in eos_ids
    return dict(generated_tokens=len(ids), generated_tokens_excluding_terminal_eos=len(ids)-int(eos),
                ttft_generate_seconds=times[0]-started,
                first_to_last_token_seconds=decode,
                decode_tokens_per_second=(len(ids)-1)/decode if len(ids)>1 and decode>0 else None,
                generation_seconds=finished-started,
                generation_tokens_per_second=len(ids)/(finished-started) if finished>started else None,
                termination="eos" if eos else "max_new_tokens" if len(ids)>=budget else "other",
                hit_output_limit=len(ids)>=budget,
                truncated_by_limit=(not eos and len(ids)>=budget))


def html_text(raw):
    """Remove exactly one enclosing Markdown fence; preserve raw output too."""
    match = re.fullmatch(r"\s*```(?:html)?\s*\n(.*?)\n```\s*", raw, flags=re.DOTALL | re.IGNORECASE)
    return (match.group(1), True) if match else (raw, False)


def check_adapter(run, lock):
    run = Path(run)
    s, recipe = read_json(run/"summary.json"), read_json(run/"recipe.json")
    if s.get("status") != "completed" or s.get("smoke"):
        raise ValueError("Inference evaluation requires a completed non-smoke run")
    if s.get("arm") not in ARMS or recipe.get("arm") != s["arm"]:
        raise ValueError("Adapter arm mismatch")
    if recipe["model_id"] != lock["model_id"] or recipe["model_revision"] != lock["revision"]:
        raise ValueError("Adapter/base-model identity mismatch")
    if recipe.get("training_method") != "bf16_lora":
        raise ValueError("Expected the 7B bf16 LoRA training patch")
    saved = read_json(run/"final_adapter/training_recipe.json")
    if digest(saved) != recipe["recipe_hash"] or s["recipe_hash"] != recipe["recipe_hash"]:
        raise ValueError("Training recipe digest mismatch")
    if any(saved.get(k) != v for k,v in recipe.items() if k not in {"recipe_hash", "environment"}):
        raise ValueError("Recipe metadata changed")
    hashes = read_json(run/"final_adapter_hashes.json")
    required = {"adapter_model.safetensors", "adapter_config.json", "training_recipe.json"}
    if not required <= hashes.keys():
        raise ValueError("Incomplete final adapter manifest")
    for name, expected in hashes.items():
        if file_sha(portable_path(run/"final_adapter", name)) != expected:
            raise ValueError(f"Adapter file changed: {name}")
    if s.get("final_adapter_digest") == s.get("initial_adapter_digest"):
        raise ValueError("Adapter was unchanged during training")
    return recipe, hashes


def held_out(bundle, recipe, limit, seed):
    meta = read_json(bundle/"bundle.json")
    rows = read_jsonl(bundle/"pages.jsonl")
    if not meta.get("ready") or meta["pages"] != len(rows) or file_sha(bundle/"pages.jsonl") != meta["pages_sha256"]:
        raise ValueError("Incomplete/changed held-out bundle")
    errors = validate_bundle_rows(bundle, rows)
    if errors or not rows:
        raise ValueError("Invalid held-out bundle: " + "; ".join(errors))
    seen = set(recipe["training_page_ids"]) | set(recipe.get("ancestor_page_ids", []))
    if seen.intersection(r["page_id"] for r in rows):
        raise ValueError("Held-out IDs overlap current/ancestor training data")
    rows = sorted(rows, key=lambda r:r["page_id"])
    random.Random(seed).shuffle(rows)
    return rows[:limit] if limit else rows, meta


def quantile(values, q):
    x = sorted(values)
    if not x:
        return None
    pos = (len(x)-1)*q
    lo = int(pos)
    return x[lo]+(x[min(lo+1,len(x)-1)]-x[lo])*(pos-lo)


def summarize(rows):
    good = [r for r in rows if r["status"] == "completed"]
    result = dict(attempts=len(rows), completed=len(good), failures=len(rows)-len(good),
                  truncated_outputs=sum(r["truncated_by_limit"] for r in good))
    for k in ["request_seconds", "generation_seconds", "ttft_generate_seconds", "ttft_request_seconds",
              "generated_tokens", "peak_allocated_bytes", "peak_reserved_bytes"]:
        vals = [r[k] for r in good]
        result[k] = dict(mean=sum(vals)/len(vals) if vals else None,
                         p50=quantile(vals,.5), p95=quantile(vals,.95))
    elapsed = sum(r["request_seconds"] for r in good)
    result["serial_pages_per_second"] = len(good)/elapsed if elapsed else None
    result["generated_tokens_total"] = sum(r["generated_tokens"] for r in good)
    return result


def compare_profiles(paths, output):
    profiles = [read_json(Path(x)/"profile.json") for x in paths]
    if len(profiles)!=3 or {s["arm"] for s in profiles}!=set(ARMS):
        raise ValueError("Supply exactly one profile directory for each arm")
    if any(not s.get("eligible_for_complete_comparison") for s in profiles):
        raise ValueError("All inference profiles must complete")
    if len({s["comparison_key"] for s in profiles})!=1:
        raise ValueError("Inference settings/data/training conditions/hardware/software differ")
    by_arm = {}
    for path,s in zip(paths,profiles):
        rows = read_jsonl(Path(path)/"requests.jsonl")
        index = {(r["repeat"],r["page_id"]):r for r in rows}
        if len(index)!=len(rows) or any(r["status"]!="completed" for r in rows):
            raise ValueError("Duplicate/failed inference requests")
        expected = {(repeat,pid) for repeat in range(s["arguments"]["repeats"]) for pid in s["page_ids"]}
        if set(index)!=expected or file_sha(Path(path)/"requests.jsonl")!=s["requests_sha256"]:
            raise ValueError("Inference request ledger incomplete or changed")
        by_arm[s["arm"]] = index
    reference = by_arm["original"]
    paired = []
    metrics = ["request_seconds","generation_seconds","ttft_generate_seconds","generated_tokens","peak_allocated_bytes"]
    for arm in ["naive","verified"]:
        candidate = by_arm[arm]
        if set(candidate)!=set(reference):
            raise ValueError("Page/repeat cohorts differ")
        for key,b in reference.items():
            v = candidate[key]
            if any(b[k]!=v[k] for k in ["image_sha256","prompt_tokens","image_tokens"]):
                raise ValueError("Paired input mismatch")
            for metric in metrics:
                paired.append(dict(arm=arm,repeat=key[0],page_id=key[1],metric=metric,
                    original=b[metric],observed=v[metric],saving_pct=100*(b[metric]-v[metric])/b[metric] if b[metric] else None,
                    either_truncated=b["truncated_by_limit"] or v["truncated_by_limit"]))
    out = Path(output)
    out.mkdir(parents=True,exist_ok=False)
    write_csv(out/"paired_inference.csv",paired)
    totals = []
    for arm in ["naive","verified"]:
        for metric in metrics:
            b = sum(r[metric] for r in reference.values())
            v = sum(r[metric] for r in by_arm[arm].values())
            totals.append(dict(arm=arm,metric=metric,aggregation="mean" if metric=="peak_allocated_bytes" else "sum",
                          original=b/len(reference) if metric=="peak_allocated_bytes" else b,
                          observed=v/len(reference) if metric=="peak_allocated_bytes" else v,
                          saving_pct=100*(b-v)/b if b else None))
    write_csv(out/"inference_savings.csv",totals)
    write_json(out/"comparison.json",dict(profiles=paths,comparison_key=profiles[0]["comparison_key"],
        quality_evaluated=False,note="Truncated and low-quality generations remain included; join reconstruction quality before claiming efficiency. Repeats are not training seeds."))
    print(f"Compared profiles: {out}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ["run", "bundle", "model-lock"]:
        p.add_argument("--"+key)
    p.add_argument("--out",required=True)
    p.add_argument("--compare",nargs=3,metavar="PROFILE_DIR")
    p.add_argument("--limit", type=int, default=0, help="0 = all held-out pages")
    p.add_argument("--order-seed", type=int, default=42)
    p.add_argument("--repeats", type=int, default=1, help="Timing repeats, not independent training seeds")
    p.add_argument("--warmup-requests", type=int, default=1)
    p.add_argument("--warmup-tokens", type=int, default=32)
    p.add_argument("--max-new-tokens", type=int, default=4096)
    p.add_argument("--gpu-hourly-rate", type=float)
    p.add_argument("--currency", default="USD")
    a = p.parse_args()
    if a.compare:
        compare_profiles(a.compare,a.out)
        return
    if not all([a.run,a.bundle,a.model_lock]):
        p.error("Generation requires --run, --bundle and --model-lock")
    if a.limit<0 or a.repeats<1 or a.warmup_requests<0 or a.warmup_tokens<1 or a.max_new_tokens<1:
        p.error("Invalid count/budget")
    if a.gpu_hourly_rate is not None and (not math.isfinite(a.gpu_hourly_rate) or a.gpu_hourly_rate<0):
        p.error("Hourly rate must be finite and nonnegative")
    out = Path(a.out).resolve()
    if out.exists():
        raise ValueError("Use a new output directory; partial runs are preserved")
    # No remote Hub access. Full model/adapter hashes checked before loading.
    lock = read_json(a.model_lock)
    recipe, adapter_hashes = check_adapter(a.run, lock)
    model_dir = verify_model_lock(lock)
    config = recipe["config"]
    bundle = Path(a.bundle).resolve()
    rows, bundle_meta = held_out(bundle, recipe, a.limit, a.order_seed)
    import torch
    from PIL import Image
    from peft import PeftModel
    from transformers import GenerationConfig, Qwen2_5_VLForConditionalGeneration
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Run on a CUDA GPU with bf16 support, using the training environment")
    torch.cuda.set_device(0)
    torch.manual_seed(a.order_seed)
    torch.cuda.manual_seed_all(a.order_seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    out.mkdir(parents=True)
    (out/"generated").mkdir()
    provenance = dict(schema=1, status="running", arm=recipe["arm"], training_recipe_hash=recipe["recipe_hash"],
                      training_seed=config["seed"], run=str(Path(a.run).resolve()),
                      adapter_hashes=adapter_hashes, model_revision=lock["revision"],
                      bundle_sha256=file_sha(bundle/"pages.jsonl"), page_ids=[r["page_id"] for r in rows],
                      preprocessing=preprocessing_contract(config), packages=package_versions(),
                      gpu_name=torch.cuda.get_device_name(0), gpu_uuid=str(getattr(torch.cuda.get_device_properties(0),"uuid","unknown")),
                      cuda_version=torch.version.cuda, arguments=vars(a),
                      code_hashes={"18_inference_profile.py":file_sha(__file__),
                                   "sft_core.py":file_sha(Path(__file__).resolve().parent.parent/"sft_core.py")},
                      leakage_check="ID disjointness including ancestor IDs; does not detect renamed or near-duplicate pages")
    write_json(out/"profile.json", provenance)
    completed_rows = []
    job_start = time.perf_counter()
    try:
        start = time.perf_counter()
        processor = load_processor(model_dir, config)
        base = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            str(model_dir), local_files_only=True, trust_remote_code=False,
            torch_dtype=torch.bfloat16, device_map={"":0}, attn_implementation="sdpa")
        model = PeftModel.from_pretrained(base, str(Path(a.run)/"final_adapter"),
                                          is_trainable=False, local_files_only=True)
        model.eval()
        model.requires_grad_(False)
        model.config.use_cache = True
        torch.cuda.synchronize()
        provenance["model_and_processor_load_seconds"] = time.perf_counter()-start
        eos = model.generation_config.eos_token_id
        eos_ids = [eos] if isinstance(eos,int) else list(eos or [])
        if not eos_ids:
            raise ValueError("No EOS token configured")
        pad_id = processor.tokenizer.pad_token_id
        generation = GenerationConfig(do_sample=False, num_beams=1, num_return_sequences=1,
                    use_cache=True, eos_token_id=eos_ids, pad_token_id=pad_id if pad_id is not None else eos_ids[0],
                    bos_token_id=model.generation_config.bos_token_id, repetition_penalty=1.0)
        provenance["generation_config"] = generation.to_dict()
        provenance["comparison_key"] = digest(dict(
            training_config=config, training_data=recipe["data_fingerprint"],
            training_code=recipe["code_hashes"],parent=recipe.get("parent_comparison_key"),
            revision=lock["revision"],eval_bundle=provenance["bundle_sha256"],
            page_ids=provenance["page_ids"],packages=provenance["packages"],
            code=provenance["code_hashes"],gpu=provenance["gpu_name"],cuda=provenance["cuda_version"],
            generation=generation.to_dict(),max_new_tokens=a.max_new_tokens,
            repeats=a.repeats,order_seed=a.order_seed,warmup_requests=a.warmup_requests,
            warmup_tokens=a.warmup_tokens,attention="sdpa",dtype="bf16",tf32=False))
        context_limit = getattr(getattr(base.config,"text_config",base.config),"max_position_embeddings",None)
        if not context_limit:
            raise ValueError("Cannot establish model context capacity")
        image_id = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")

        def generate_one(row, budget):
            # Inference peak is absolute process allocator occupancy, including resident model.
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            request_start = time.perf_counter()
            with Image.open(portable_path(bundle,row["image"])) as opened:
                im = opened.convert("RGB")
            try:
                enc = encode_image_text(processor, im, conversation_text(processor))
            finally:
                im.close()
            prompt_ids = enc["input_ids"][0].tolist()
            if len(prompt_ids)+budget > context_limit:
                raise ValueError("Prompt + output budget exceeds model context; no silent truncation")
            prep_end = time.perf_counter()
            enc = enc.to("cuda:0")
            torch.cuda.synchronize()
            transfer_end = time.perf_counter()
            timer = TokenClock(prompt_ids)
            gen_start = time.perf_counter()
            with torch.inference_mode():
                output = model.generate(**enc, generation_config=generation, max_new_tokens=budget,
                                        streamer=timer, return_dict_in_generate=False)
            torch.cuda.synchronize()
            gen_end = time.perf_counter()
            ids = output[0,len(prompt_ids):].cpu().tolist()
            if ids != timer.ids:
                raise ValueError("Streamer token IDs disagree with generated output")
            raw = processor.tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            request_end = time.perf_counter()
            cleaned, stripped = html_text(raw)
            record = token_metrics(ids,timer.times,gen_start,gen_end,eos_ids,budget) | dict(
                prompt_tokens=len(prompt_ids), image_tokens=prompt_ids.count(image_id),
                vision_patch_tokens=int(enc["image_grid_thw"].prod(dim=-1).sum().item()),
                cpu_preparation_seconds=prep_end-request_start,
                transfer_seconds=transfer_end-prep_end,
                request_seconds=request_end-request_start,
                ttft_request_seconds=timer.times[0]-request_start,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                markdown_fence_removed=stripped, empty_output=not cleaned.strip(),
                output_exceeds_training_sequence_cap=len(prompt_ids)+len(ids)>config["max_seq_length"])
            del output, enc
            return record, raw, cleaned, ids

        warm_start = time.perf_counter()
        for i in range(a.warmup_requests):
            generate_one(rows[i%len(rows)], min(a.warmup_tokens,a.max_new_tokens))
        provenance["warmup_seconds"] = time.perf_counter()-warm_start
        # Clear unused warmup cache once; never empty the cache between measured requests.
        gc.collect()
        torch.cuda.empty_cache()
        for repeat in range(a.repeats):
            for i,row in enumerate(rows):
                identity = dict(page_id=row["page_id"], image_sha256=row["image_sha256"],
                                arm=recipe["arm"], repeat=repeat, page_index=i)
                try:
                    metric, raw, cleaned, ids = generate_one(row,a.max_new_tokens)
                    stem = f"r{repeat:02d}_p{i:06d}"
                    (out/"generated"/(stem+".raw.txt")).write_text(raw,encoding="utf-8",newline="\n")
                    (out/"generated"/(stem+".html")).write_text(cleaned,encoding="utf-8",newline="\n")
                    write_json(out/"generated"/(stem+".tokens.json"),ids)
                    result = identity | metric | dict(status="completed", html=f"generated/{stem}.html",
                                      original_reference=row["original"], verified_reference=row["verified"])
                    append_jsonl(out/"requests.jsonl",result)
                    completed_rows.append(result)
                    print(f"{recipe['arm']} {repeat+1}/{a.repeats} {i+1}/{len(rows)} {row['page_id']}: {metric['generated_tokens']} tokens, {metric['generation_seconds']:.2f}s",flush=True)
                except BaseException as exc:
                    result = identity | dict(status="failed",error=repr(exc))
                    append_jsonl(out/"requests.jsonl",result)
                    completed_rows.append(result)
                    raise  # OOM/other errors fail the run, never cherry-pick successful pages.
        provenance["status"] = "completed"
    except BaseException as exc:
        provenance.update(status="failed",error=repr(exc))
        raise
    finally:
        provenance["profile_job_seconds"] = time.perf_counter()-job_start
        provenance["aggregates"] = summarize(completed_rows)
        provenance["eligible_for_complete_comparison"] = provenance["status"]=="completed"
        if (out/"requests.jsonl").exists():
            provenance["requests_sha256"] = file_sha(out/"requests.jsonl")
        if a.gpu_hourly_rate is not None and provenance["status"]=="completed":
            n = len(completed_rows)
            provenance["estimated_request_cost_per_1000"] = sum(r["request_seconds"] for r in completed_rows)/n*1000*a.gpu_hourly_rate/3600
            provenance["estimated_profile_job_cost"] = provenance["profile_job_seconds"]*a.gpu_hourly_rate/3600
            provenance["currency"] = a.currency
        write_csv(out/"requests.csv",completed_rows)
        write_json(out/"profile.json",provenance)
    print(f"COMPLETE: {out}. Quality evaluation of generated HTML is still required.")


if __name__ == "__main__":
    main()
