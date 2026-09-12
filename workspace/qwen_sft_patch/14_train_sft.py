"""Single-GPU Qwen2.5-VL bf16 LoRA SFT. Training only; no test evaluation or inference."""
from __future__ import annotations
import argparse
import datetime as dt
import hashlib
import math
import os
import platform
import random
import subprocess
import time
import traceback
from pathlib import Path
from sft_core import *
from telemetry import Telemetry

TOKEN_KEYS = ["sequence_tokens", "target_tokens", "image_tokens", "prompt_nonimage_tokens", "vision_patch_tokens", "padding_tokens"]

def adapter_digest(model):
    import torch
    h = hashlib.sha256()
    for name, p in sorted(model.named_parameters()):
        if p.requires_grad:
            h.update(name.encode())
            h.update(p.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes())
    return h.hexdigest()

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--prepared", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--model-lock", required=True)
    ap.add_argument("--arm", choices=ARMS, required=True)
    ap.add_argument("--out", required=True, help="New session directory, including for resume")
    ap.add_argument("--resume", help="Own trusted checkpoint directory; loads Python optimizer/RNG state")
    ap.add_argument("--init-from-run", help="Completed same-arm run; load its adapter with a NEW optimizer/schedule for disjoint new pages")
    ap.add_argument("--smoke", action="store_true", help="Train two diagnostic updates on worst-size examples; excluded from research comparisons")
    args = ap.parse_args()
    if args.init_from_run and (args.resume or args.smoke):
        ap.error("Continuation cannot combine with resume or smoke")
    if args.smoke and args.resume:
        ap.error("Smoke tests cannot resume")
    out = Path(args.out).resolve()
    if out.exists():
        ap.error("Output exists; choose a new directory")
    out.mkdir(parents=True)
    started = time.perf_counter()
    summary = {"status": "starting", "arm": args.arm, "smoke": args.smoke,
               "resumed": bool(args.resume), "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
               "session_output": str(out), "completed_updates_this_session": 0}
    write_json(out / "summary.json", summary)
    monitor = None
    try:
        import numpy as np
        import torch
        from PIL import Image
        from transformers import Qwen2_5_VLForConditionalGeneration, get_cosine_schedule_with_warmup
        from peft import LoraConfig, TaskType, get_peft_model, PeftModel
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise ValueError("Exactly one visible CUDA GPU required")
        if not torch.cuda.is_bf16_supported():
            raise ValueError("BF16-capable GPU required")
        c, lock = load_config(args.config), read_json(args.model_lock)
        data = Path(args.prepared).resolve()
        prepared = read_json(data / "prepared.json")
        if not prepared.get("ready") or prepared["preprocessing"] != preprocessing_contract(c):
            raise ValueError("Dataset not ready, or preprocessing config differs: re-prepare all arms")
        if prepared["model_revision"] != lock["revision"]:
            raise ValueError("Prepared data and model lock use different revisions")
        if prepared["packages"] != package_versions():
            raise ValueError("Installed package versions changed since preparation; use one pinned environment")
        model_dir = verify_model_lock(lock)
        if file_sha(data / f"{args.arm}.jsonl") != prepared["manifest_hashes"][args.arm]:
            raise ValueError("Prepared arm manifest changed")
        rows = read_jsonl(data / f"{args.arm}.jsonl")
        if [r["page_id"] for r in rows] != prepared["page_ids"] or not rows:
            raise ValueError("Arm IDs differ from frozen paired training population")
        errors = []
        for r in rows:
            for key in ["html", "image"]:
                try:
                    if file_sha(portable_path(data, r[key])) != r[key+"_sha256"]:
                        errors.append(f"{r['page_id']}: changed {key}")
                except Exception as e:
                    errors.append(f"{r['page_id']}: {e}")
        if errors:
            raise ValueError("\n".join(errors))
        props = torch.cuda.get_device_properties(0)
        environment = {"gpu": props.name, "vram_bytes": props.total_memory,
                       "gpu_uuid": str(getattr(props, "uuid", "unavailable")),
                       "cuda_runtime": torch.version.cuda, "python": platform.python_version(),
                       "platform": platform.platform(), "packages": package_versions(),
                       "attention": "sdpa", "tf32": False, "microbatch_size": 1,
                       "packing": False, "gradient_checkpointing": "non-reentrant"}
        try:
            environment["nvidia_driver"] = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], text=True).strip()
        except Exception:
            environment["nvidia_driver"] = None
        parent = validate_parent_run(args.init_from_run, args.arm, c, lock, prepared["page_ids"]) if args.init_from_run else None
        if args.resume:
            # Preserve the stage's ancestry when resuming an interrupted continuation.
            resume_recipe = read_json(Path(args.resume) / "stage_recipe.json")
            parent = resume_recipe.get("parent_run")
        recipe = {"training_method": "bf16_lora", "parent_run": parent,
                  "parent_comparison_key": parent["comparison_key"] if parent else None,
                  "training_page_ids": prepared["page_ids"],
                  "ancestor_page_ids": parent["ancestor_page_ids"] if parent else [], "config": c, "data_fingerprint": prepared["data_fingerprint"],
                  "model_id": lock["model_id"], "model_revision": lock["revision"],
                  "arm": args.arm, "code_hashes": source_hashes(), "packages": package_versions()}
        recipe_hash = digest(recipe)
        write_json(out / "recipe.json", recipe | {"recipe_hash": recipe_hash, "environment": environment})
        summary.update({"environment": environment, "config": c, "data_fingerprint": prepared["data_fingerprint"],
                        "model_revision": lock["revision"], "recipe_hash": recipe_hash,
                        "parent_comparison_key": recipe["parent_comparison_key"], "parent_run": parent,
                        "code_hashes": recipe["code_hashes"], "pages": len(rows),
                        "shared_preparation_seconds": prepared["preparation_seconds"]})
        monitor = Telemetry(out / "telemetry.jsonl", environment["gpu_uuid"])
        monitor.start()
        random.seed(c["seed"])
        np.random.seed(c["seed"])
        torch.manual_seed(c["seed"])
        torch.cuda.manual_seed_all(c["seed"])
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        processor = load_processor(model_dir, c)
        image_id = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
        summary["preflight_seconds"] = time.perf_counter()-started
        monitor.phase = "model_load"
        torch.cuda.reset_peak_memory_stats()
        load_start = time.perf_counter()
        base = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            str(model_dir), local_files_only=True, trust_remote_code=False,
            torch_dtype=torch.bfloat16, device_map={"": 0}, attn_implementation="sdpa")
        base.config.use_cache = False
        base.requires_grad_(False)
        base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        targets = allowed_lora_targets([name for name, module in base.named_modules()])
        if args.resume:
            cp = Path(args.resume).resolve()
            prior = read_json(cp / "checkpoint.json")
            if prior["recipe_hash"] != recipe_hash:
                raise ValueError("Resume recipe differs (data/model/config/code/environment packages); use original recipe")
            if file_sha(cp / "training_state.pt") != prior["state_sha256"]:
                raise ValueError("Checkpoint state hash mismatch")
            if file_sha(cp / "adapter_model.safetensors") != prior["adapter_sha256"]:
                raise ValueError("Checkpoint adapter hash mismatch")
            model = PeftModel.from_pretrained(base, str(cp), is_trainable=True)
        elif parent:
            model = PeftModel.from_pretrained(base, parent["adapter_path"], is_trainable=True)
        else:
            model = get_peft_model(base, LoraConfig(task_type=TaskType.CAUSAL_LM,
                r=c["lora_r"], lora_alpha=c["lora_alpha"], lora_dropout=c["lora_dropout"],
                target_modules=targets, bias="none"))
        trainable = [(n,p) for n,p in model.named_parameters() if p.requires_grad]
        if not trainable or any("lora_" not in n or ".visual." in n for n,p in trainable):
            raise ValueError("Unexpected trainable parameters; only language-decoder LoRA adapters are allowed")
        if any(p.device.type != "cuda" for p in model.parameters()):
            raise ValueError("CPU/offloaded parameters found; this recipe requires all model parameters on one GPU")
        write_json(out / "trainable_parameters.json", {"target_modules": targets,
                   "names": [n for n,p in trainable], "trainable_numel": sum(p.numel() for n,p in trainable),
                   "all_numel_reported_by_torch": sum(p.numel() for p in model.parameters())})
        torch.cuda.synchronize()
        summary["model_load_seconds"] = time.perf_counter()-load_start
        summary["model_load_peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
        summary["model_load_peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
        # Smoke covers long text, many image patches, and high combined sequence*vision load.
        smoke_indices = list(dict.fromkeys(max(range(len(rows)), key=lambda i: rows[i][key])
                                          for key in ["sequence_tokens", "vision_patch_tokens"]))
        cost_index = max(range(len(rows)), key=lambda i: rows[i]["sequence_tokens"]*rows[i]["vision_patch_tokens"])
        smoke_indices = list(dict.fromkeys(smoke_indices+[cost_index]))
        accumulation = c["gradient_accumulation_steps"]
        if args.smoke:
            windows = [(0, 0, (smoke_indices*accumulation)[:accumulation]),
                       (0, accumulation, (smoke_indices[::-1]*accumulation)[:accumulation])]
            total_updates = 2
        else:
            total_updates = math.ceil(len(rows)/accumulation)*c["epochs"]
            windows = [(epoch, start, order[start:start+accumulation])
                       for epoch in range(c["epochs"])
                       for order in [epoch_order(len(rows), c["seed"], epoch)]
                       for start in range(0, len(rows), accumulation)]
        optimizer = torch.optim.AdamW([p for n,p in trainable], lr=c["learning_rate"],
                                     weight_decay=c["weight_decay"], foreach=False)
        scheduler = get_cosine_schedule_with_warmup(optimizer, math.ceil(total_updates*c["warmup_ratio"]), total_updates)
        state = {"global_step": 0, "next_window": 0, "logical_examples": 0,
                 "logical_tokens": {k: 0 for k in TOKEN_KEYS}, "initial_adapter_digest": adapter_digest(model)}
        if args.resume:
            # Only load checkpoints produced by this patch and owned by you: optimizer/RNG state uses pickle.
            saved = torch.load(cp / "training_state.pt", map_location="cpu", weights_only=False)
            optimizer.load_state_dict(saved["optimizer"])
            scheduler.load_state_dict(saved["scheduler"])
            state = saved["progress"]
            random.setstate(saved["python_rng"])
            np.random.set_state(saved["numpy_rng"])
            torch.set_rng_state(saved["torch_rng"])
            torch.cuda.set_rng_state_all(saved["cuda_rng"])
        if state["next_window"] >= len(windows):
            raise ValueError("Checkpoint already completed all updates; its adapter is the completed training state")
        checkpoint_seconds = 0.0
        peak_allocated = peak_reserved = 0
        session_tokens = {k: 0 for k in TOKEN_KEYS}
        session_examples = 0
        summary["completed_backwards_this_session"] = 0

        def checkpoint():
            nonlocal checkpoint_seconds
            monitor.phase = "checkpoint"
            torch.cuda.synchronize()
            before = time.perf_counter()
            folder = out / "checkpoints" / f"step_{state['global_step']:06d}"
            if folder.exists():
                return
            temp = folder.with_name(folder.name + ".incomplete")
            temp.mkdir(parents=True)
            model.save_pretrained(temp, safe_serialization=True)
            write_json(temp / "stage_recipe.json", recipe)
            torch.save({"progress": state, "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                        "python_rng": random.getstate(), "numpy_rng": np.random.get_state(),
                        "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all()}, temp / "training_state.pt")
            write_json(temp / "checkpoint.json", {"recipe_hash": recipe_hash,
                       "state_sha256": file_sha(temp / "training_state.pt"),
                       "adapter_sha256": file_sha(temp / "adapter_model.safetensors"), "progress": state})
            temp.rename(folder)
            write_json(out / "latest_checkpoint.json", {"path": str(folder)})
            checkpoint_seconds += time.perf_counter()-before
            monitor.phase = "train"

        model.train()
        loop_start = time.perf_counter()
        for window_index in range(state["next_window"], len(windows)):
            epoch, offset, indices = windows[window_index]
            monitor.phase = "train"
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            begin = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            weights = loss_weights([rows[i]["target_tokens"] for i in indices], c["loss_normalization"])
            counts = {k: 0 for k in TOKEN_KEYS}
            loss_value = torch.zeros((), device="cuda")
            preparation_seconds = 0.0
            for index, weight in zip(indices, weights):
                r = rows[index]
                prep_start = time.perf_counter()
                with Image.open(portable_path(data, r["image"])) as opened:
                    im = opened.convert("RGB")
                try:
                    html = portable_path(data, r["html"]).read_text(encoding="utf-8")
                    enc = encode_image_text(processor, im, conversation_text(processor, html))
                finally:
                    im.close()
                ids = enc["input_ids"][0].tolist()
                if digest(ids) != r["input_ids_sha256"] or enc["image_grid_thw"].tolist() != r["image_grid_thw"]:
                    raise ValueError(f"Processor output changed for {r['page_id']}")
                if len(ids) > c["max_seq_length"]:
                    raise ValueError("Sequence exceeds cap; no truncation is permitted")
                labels = torch.tensor([mask_suffix(ids, ids[:r["prefix_tokens"]], image_id)], dtype=torch.long)
                if int((labels[:,1:] != -100).sum()) != r["target_tokens"]:
                    raise ValueError("Actual supervised-token count differs from preparation")
                preparation_seconds += time.perf_counter()-prep_start
                enc = {k: v.to("cuda") for k,v in enc.items()}
                labels = labels.to("cuda")
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    result = model(**enc, labels=labels, use_cache=False)
                    weighted_loss = result.loss*weight
                weighted_loss.backward()
                loss_value += weighted_loss.detach()
                for key in TOKEN_KEYS:
                    counts[key] += r[key]
                summary["completed_backwards_this_session"] += 1
                append_jsonl(out / "microbatches.jsonl", {"pending_update": state["global_step"]+1,
                    "epoch": epoch+1, "page_id": r["page_id"], **{k:r[k] for k in TOKEN_KEYS}})
                del result, weighted_loss, enc, labels
            grad_norm = torch.nn.utils.clip_grad_norm_([p for n,p in trainable], c["max_grad_norm"], error_if_nonfinite=True)
            if not torch.isfinite(loss_value).item():
                raise ValueError("Nonfinite training loss")
            learning_rate = optimizer.param_groups[0]["lr"]
            optimizer.step()
            scheduler.step()
            torch.cuda.synchronize()
            wall = time.perf_counter()-begin
            allocated, reserved = torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved()
            peak_allocated, peak_reserved = max(peak_allocated, allocated), max(peak_reserved, reserved)
            state["global_step"] += 1
            state["next_window"] = window_index+1
            state["logical_examples"] += len(indices)
            session_examples += len(indices)
            for key in TOKEN_KEYS:
                state["logical_tokens"][key] += counts[key]
                session_tokens[key] += counts[key]
            record = {"global_step": state["global_step"], "epoch": epoch+1, "examples": len(indices),
                      "page_ids": [rows[i]["page_id"] for i in indices], **counts,
                      "update_wall_seconds": wall, "cpu_preparation_seconds": preparation_seconds,
                      "loss": loss_value.item(), "learning_rate": learning_rate, "grad_norm": grad_norm.item(),
                      "peak_allocated_bytes": allocated, "peak_reserved_bytes": reserved,
                      "warmup_for_timing": summary["completed_updates_this_session"] < c["warmup_metric_updates"]}
            append_jsonl(out / "steps.jsonl", record)
            summary["completed_updates_this_session"] += 1
            summary.update({"session_examples": session_examples, "session_tokens": session_tokens,
                            "progress": state, "training_peak_allocated_bytes": peak_allocated,
                            "training_peak_reserved_bytes": peak_reserved})
            write_json(out / "summary.json", summary | {"status": "training"})
            print(f"{args.arm} update {state['global_step']}/{total_updates} loss={record['loss']:.4f} wall={wall:.2f}s peak={allocated/2**30:.2f}GiB", flush=True)
            at_epoch_end = offset+len(indices) >= len(rows)
            if not args.smoke and (state["global_step"] % c["save_every_updates"] == 0 or at_epoch_end):
                checkpoint()
        summary["training_loop_wall_seconds"] = time.perf_counter()-loop_start
        summary["checkpoint_seconds"] = checkpoint_seconds
        summary["training_loop_excluding_checkpoint_seconds"] = summary["training_loop_wall_seconds"]-checkpoint_seconds
        monitor.phase = "final_save"
        save_start = time.perf_counter()
        final = out / "final_adapter"
        model.save_pretrained(final, safe_serialization=True)
        processor.save_pretrained(final)
        write_json(final / "training_recipe.json", recipe)
        final_digest = adapter_digest(model)
        if final_digest == state["initial_adapter_digest"]:
            raise ValueError("Adapter parameters did not change; training sanity check failed")
        final_hashes = {str(p.relative_to(final)): file_sha(p) for p in final.rglob("*") if p.is_file()}
        write_json(out / "final_adapter_hashes.json", final_hashes)
        summary.update({"status": "completed", "final_save_seconds": time.perf_counter()-save_start,
                        "final_adapter": str(final), "initial_adapter_digest": state["initial_adapter_digest"],
                        "final_adapter_digest": final_digest, "expected_total_updates": total_updates,
                        "checkpoint_policy": "fixed final epoch; no validation-loss selection"})
    except BaseException as e:
        summary.update({"status": "interrupted" if isinstance(e, KeyboardInterrupt) else "failed",
                        "error": repr(e), "traceback": traceback.format_exc()})
        raise
    finally:
        if monitor:
            try:
                summary["telemetry"] = monitor.stop()
            except Exception as e:
                summary["telemetry_error"] = repr(e)
        summary["job_wall_seconds"] = time.perf_counter()-started
        summary["timing_scope"] = "script after argument parsing: preflight/imports/hash checks, load, train, logging, checkpoints, final adapter save; excludes installation/download/dataset preparation"
        write_json(out / "summary.json", summary)

if __name__ == "__main__":
    main()
