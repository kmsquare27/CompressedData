"""Shared, mostly standard-library helpers. No training work at import time."""
from __future__ import annotations
import csv
import hashlib
import json
import math
import os
import random
import re
from pathlib import Path

MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
ARMS = ("original", "naive", "verified")
PROMPT = (
    "Generate the HTML code for the supplied webpage screenshot. "
    "Match the visible content and layout. Return only "
    "the HTML code, without Markdown fences or an explanation."
)

def clean_json(x):
    if isinstance(x, float) and not math.isfinite(x):
        return None
    if isinstance(x, dict):
        return {str(k): clean_json(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean_json(v) for v in x]
    if isinstance(x, Path):
        return str(x)
    if hasattr(x, "item"):
        return clean_json(x.item())
    return x

def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(clean_json(value), indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    os.replace(tmp, path)

def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

def digest(value):
    return hashlib.sha256(json.dumps(clean_json(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()

def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for part in iter(lambda: f.read(1024*1024), b""):
            h.update(part)
    return h.hexdigest()

def write_jsonl(path, rows):
    with Path(path).open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(clean_json(row), ensure_ascii=False, allow_nan=False) + "\n")

def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

def append_jsonl(path, row):
    with Path(path).open("a", encoding="utf-8") as f:
        f.write(json.dumps(clean_json(row), ensure_ascii=False, allow_nan=False) + "\n")
        f.flush()

def write_csv(path, rows):
    cols = list(dict.fromkeys(k for r in rows for k in r))
    with Path(path).open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

def read_csv(path, required=True):
    p = Path(path)
    if not p.is_file():
        if required:
            raise ValueError(f"Missing file: {p}")
        return []
    with p.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))

def truth(v):
    return str(v).strip().lower() in {"true", "1", "1.0", "yes"}

def nonempty(v):
    return v is not None and str(v).strip().lower() not in {"", "nan", "none"}

def unique_index(rows, label, errors):
    out = {}
    for row in rows:
        pid = row.get("page_id", "").strip()
        if not pid or pid in out:
            errors.append(f"{label}: blank/duplicate page_id {pid!r}")
        out[pid] = row
    return out

def portable_path(base, relative):
    """Bundle paths must stay inside the bundle, including after symlink resolution."""
    base = Path(base).resolve()
    p = (base / relative).resolve()
    if not p.is_relative_to(base):
        raise ValueError(f"Path escapes bundle: {relative}")
    return p

def load_config(path):
    c = read_json(path)
    required = {"model_id", "epochs", "seed", "gradient_accumulation_steps", "learning_rate",
                "weight_decay", "warmup_ratio", "loss_normalization", "lora_r", "lora_alpha",
                "lora_dropout", "max_seq_length", "min_image_tokens", "max_image_tokens",
                "save_every_updates", "warmup_metric_updates", "max_grad_norm"}
    missing = sorted(required - c.keys())
    if missing:
        raise ValueError(f"Configuration missing: {missing}")
    if c["model_id"] != MODEL_ID:
        raise ValueError(f"This implementation targets {MODEL_ID}; a different architecture needs a separate validated patch")
    if c["loss_normalization"] not in {"example", "token"}:
        raise ValueError("loss_normalization must be example or token")
    for k in ["epochs", "gradient_accumulation_steps", "max_seq_length", "min_image_tokens", "max_image_tokens", "lora_r", "save_every_updates"]:
        if not isinstance(c[k], int) or c[k] < 1:
            raise ValueError(f"{k} must be a positive integer")
    if c["min_image_tokens"] > c["max_image_tokens"]:
        raise ValueError("min_image_tokens exceeds max_image_tokens")
    if not 0 <= c["warmup_ratio"] < 1 or not 0 <= c["lora_dropout"] < 1:
        raise ValueError("Invalid warmup_ratio/lora_dropout")
    if c["learning_rate"] <= 0 or c["weight_decay"] < 0 or c["max_grad_norm"] <= 0 or c["lora_alpha"] <= 0:
        raise ValueError("Invalid optimizer/LoRA numeric parameter")
    if not isinstance(c["warmup_metric_updates"], int) or c["warmup_metric_updates"] < 0:
        raise ValueError("warmup_metric_updates must be a nonnegative integer")
    return c

def preprocessing_contract(c):
    return {k: c[k] for k in ["model_id", "max_seq_length", "min_image_tokens", "max_image_tokens"]} | {"prompt": PROMPT, "image_processor_use_fast": False, "tokenizer_use_fast": True, "padding": False, "truncation": False}

def user_message():
    return {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": PROMPT}]}

def conversation_text(processor, html=None):
    messages = [user_message()]
    if html is not None:
        messages.append({"role": "assistant", "content": [{"type": "text", "text": html}]})
    return processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=html is None)

def mask_suffix(full_ids, prefix_ids, image_token_id):
    """Never infer assistant labels by searching arbitrary HTML for a role string."""
    p = len(prefix_ids)
    if full_ids[:p] != prefix_ids or len(full_ids) <= p:
        raise ValueError("Chat-template prefix is not an exact token prefix of the full example")
    if image_token_id in full_ids[p:]:
        raise ValueError("Image placeholder found inside supervised assistant target")
    labels = [-100]*p + list(full_ids[p:])
    if any(v != -100 for v in labels[:p]):
        raise AssertionError("Prompt labels leaked")
    return labels

def mask_from_offsets(full_ids, offsets, boundary, text, image_token_id):
    """Handle BPE merging of template newline with leading HTML whitespace.

    A token spanning that boundary is masked in full; only whitespace may straddle.
    No HTML content token or user/image token is silently assigned to the other side.
    """
    if len(full_ids) != len(offsets):
        raise ValueError("Offset/token length mismatch")
    crossing_chars = 0
    for a,b in offsets:
        if a < boundary < b:
            if not text[a:b].isspace():
                raise ValueError("A non-whitespace token spans the assistant boundary; review this template")
            crossing_chars += b-boundary
    starts = [i for i,(a,b) in enumerate(offsets) if a >= boundary and b > a]
    if not starts:
        raise ValueError("No assistant target tokens")
    p = starts[0]
    return mask_suffix(full_ids, full_ids[:p], image_token_id), crossing_chars

def assistant_labels(processor, full_ids, prefix_ids, full_text, prefix_text, image_token_id):
    if full_ids[:len(prefix_ids)] == prefix_ids:
        return mask_suffix(full_ids, prefix_ids, image_token_id), 0
    if not full_text.startswith(prefix_text) or prefix_text.count("<|image_pad|>") != 1:
        raise ValueError("Unexpected single-image chat template")
    # The processor expands one placeholder into one token per merged visual position.
    marker = "<|image_pad|>"
    expansion = marker*full_ids.count(image_token_id)
    expanded = full_text.replace(marker, expansion)
    boundary = len(prefix_text.replace(marker, expansion))
    tokens = processor.tokenizer(expanded, add_special_tokens=False, return_offsets_mapping=True)
    if list(tokens["input_ids"]) != full_ids:
        raise ValueError("Offset tokenizer does not reproduce actual processor IDs")
    return mask_from_offsets(full_ids, tokens["offset_mapping"], boundary, expanded, image_token_id)

def loss_weights(target_counts, normalization):
    if not target_counts or min(target_counts) <= 0:
        raise ValueError("Every example must have at least one supervised token")
    if normalization == "example":
        return [1/len(target_counts)]*len(target_counts)
    if normalization == "token":
        return [n/sum(target_counts) for n in target_counts]
    raise ValueError(normalization)

def epoch_order(n, seed, epoch):
    out = list(range(n))
    random.Random(seed+epoch).shuffle(out)
    return out

def validate_bundle_rows(bundle, rows):
    errors = []
    unique_index(rows, "bundle", errors)
    for row in rows:
        for key in ["original", "verified", "image"]:
            try:
                p = portable_path(bundle, row[key])
                if not p.is_file():
                    raise ValueError(f"missing {p}")
                if file_sha(p) != row[key+"_sha256"]:
                    raise ValueError(f"changed {key} bytes")
            except Exception as e:
                errors.append(f"{row.get('page_id')}: {e}")
    return errors

def load_processor(model_dir, config):
    from transformers import AutoProcessor, AutoTokenizer
    processor = AutoProcessor.from_pretrained(
        str(model_dir), local_files_only=True, trust_remote_code=False, use_fast=False,
        min_pixels=config["min_image_tokens"]*28*28,
        max_pixels=config["max_image_tokens"]*28*28,
    )
    # Explicitly choose a fast TEXT tokenizer and the reference slow IMAGE processor.
    processor.tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True,
                                                        trust_remote_code=False, use_fast=True)
    processor.tokenizer.padding_side = "right"
    if not processor.tokenizer.is_fast:
        raise ValueError("The Qwen tokenizer must be fast and exact; no fallback is provided")
    return processor

def encode_image_text(processor, image, text):
    return processor(text=[text], images=[image], padding=False, truncation=False,
                     return_tensors="pt")

def allowed_lora_targets(module_names):
    pattern = re.compile(r".*\.layers\.\d+\.(?:self_attn\.(?:q|k|v|o)_proj|mlp\.(?:gate|up|down)_proj)$")
    targets = [n for n in module_names if pattern.fullmatch(n) and ".visual." not in n]
    if not targets:
        raise ValueError("No language-decoder LoRA targets matched; inspect installed model architecture")
    return targets

def package_versions():
    import importlib.metadata as m
    result = {}
    for name in ["torch", "torchvision", "transformers", "tokenizers", "peft", "accelerate", "Pillow", "numpy", "huggingface-hub", "safetensors", "minify-html", "psutil", "nvidia-ml-py"]:
        try:
            result[name] = m.version(name)
        except m.PackageNotFoundError:
            result[name] = None
    return result

def source_hashes():
    root = Path(__file__).resolve().parent
    return {p.name: file_sha(p) for p in sorted(root.glob("*.py"))}

def verify_model_lock(lock):
    if lock["model_id"] != MODEL_ID or not re.fullmatch(r"[0-9a-f]{40}", lock["revision"]):
        raise ValueError("Unexpected model ID/revision in lock")
    if "config.json" not in lock["files_sha256"] or not any(n.endswith(".safetensors") for n in lock["files_sha256"]):
        raise ValueError("Model lock must cover configuration and model weights")
    directory = Path(lock["snapshot_path"])
    errors = []
    # HF snapshots contain symlinks to content-addressed blobs, so do not apply bundle containment here.
    for name, expected in lock["files_sha256"].items():
        p = directory / name
        if not p.is_file() or file_sha(p) != expected:
            errors.append(f"Model snapshot missing/changed: {name}")
    if errors:
        raise ValueError("\n".join(errors))
    return directory


def validate_parent_run(run, arm, config, lock, page_ids):
    """Validate a completed, same-arm predecessor before fresh-stage optimization."""
    run = Path(run).resolve()
    summary = read_json(run / "summary.json")
    recipe = read_json(run / "recipe.json")
    if summary.get("status") != "completed" or summary.get("smoke"):
        raise ValueError("Parent must be a completed non-smoke training run")
    if recipe["arm"] != arm or summary["arm"] != arm:
        raise ValueError("Parent arm mismatch")
    if recipe["model_id"] != lock["model_id"] or recipe["model_revision"] != lock["revision"]:
        raise ValueError("Parent base model/revision mismatch")
    if recipe.get("training_method") != "bf16_lora":
        raise ValueError("Parent must use this bf16 LoRA recipe")
    for key in ("model_id", "seed", "lora_r", "lora_alpha", "lora_dropout",
                "max_seq_length", "min_image_tokens", "max_image_tokens",
                "gradient_accumulation_steps", "learning_rate", "weight_decay",
                "warmup_ratio", "loss_normalization", "max_grad_norm", "epochs"):
        if recipe["config"][key] != config[key]:
            raise ValueError(f"Parent config mismatch: {key}")
    seen = set(recipe["training_page_ids"]) | set(recipe.get("ancestor_page_ids", []))
    if seen.intersection(page_ids):
        raise ValueError("Continuation pages overlap previous training page IDs")
    folder = run / "final_adapter"
    hashes = read_json(run / "final_adapter_hashes.json")
    if not {"adapter_model.safetensors", "adapter_config.json", "training_recipe.json"} <= hashes.keys():
        raise ValueError("Incomplete parent adapter manifest")
    for name, expected in hashes.items():
        if file_sha(portable_path(folder, name)) != expected:
            raise ValueError(f"Parent artifact changed: {name}")
    saved_recipe = read_json(folder / "training_recipe.json")
    if digest(saved_recipe) != recipe["recipe_hash"]:
        raise ValueError("Parent recipe integrity mismatch")
    if summary["recipe_hash"] != recipe["recipe_hash"]:
        raise ValueError("Parent summary/recipe mismatch")
    # This key excludes arm-specific targets but keeps comparable predecessor conditions.
    group = digest({"config": recipe["config"], "data": recipe["data_fingerprint"],
                    "model": recipe["model_revision"], "code": recipe["code_hashes"],
                    "parent": recipe.get("parent_comparison_key")})
    return {"run": str(run), "adapter_path": str(folder),
            "adapter_sha256": hashes["adapter_model.safetensors"],
            "recipe_hash": recipe["recipe_hash"], "comparison_key": group,
            "ancestor_page_ids": sorted(seen)}
