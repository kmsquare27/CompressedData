#!/usr/bin/env python3
"""Apply focused SFT integrity/recovery fixes to the supplied patched project."""
from __future__ import annotations

import ast
import datetime as dt
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile
import textwrap
import zipfile

ROOT = Path(
    sys.argv[1] if len(sys.argv) > 1 else "/workspace/qwen_sft_patch"
).resolve()
if not (ROOT / "sft_core.py").is_file():
    raise SystemExit(f"SFT project not found: {ROOT}")

CHANGES = {}


def block(text):
    return textwrap.dedent(text).strip("\n") + "\n"


def current(name):
    if name in CHANGES:
        return CHANGES[name]
    path = ROOT / name
    if not path.is_file():
        raise RuntimeError(f"Required file missing: {path}")
    return path.read_text(encoding="utf-8")


def replace(name, old, new):
    source = current(name)
    count = source.count(old)
    if count != 1:
        raise RuntimeError(
            f"{name}: expected one matching source block, found {count}. "
            "No project changes have been applied."
        )
    CHANGES[name] = source.replace(old, new, 1)


def replace_function(name, function_name, replacement):
    source = current(name)
    nodes = [
        node for node in ast.parse(source).body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == function_name
    ]
    if len(nodes) != 1:
        raise RuntimeError(f"{name}: cannot locate {function_name}")
    node = nodes[0]
    if node.decorator_list:
        raise RuntimeError(f"Unexpected decorated function: {function_name}")
    lines = source.splitlines(keepends=True)
    CHANGES[name] = (
        "".join(lines[:node.lineno - 1])
        + block(replacement)
        + "".join(lines[node.end_lineno:])
    )


# ---------------------------------------------------------------------------
# Shared helpers: bounded pruning, checkpoint integrity, realistic disk budget.
# ---------------------------------------------------------------------------

CORE = "sft_core.py"

replace_function(CORE, "checkpoint_dirs", r'''
def checkpoint_dirs(run, recipe_hash=None):
    """Recognized completed checkpoints for this recipe, in numeric step order."""
    run = Path(run)
    folder = run / "checkpoints"
    if not folder.is_dir():
        return []
    if folder.is_symlink():
        raise ValueError("Checkpoint directory must not be a symlink")
    if recipe_hash is None:
        recipe_hash = read_json(run / "recipe.json")["recipe_hash"]
    found = []
    for directory in folder.iterdir():
        match = re.fullmatch(r"step_(\d+)", directory.name)
        if not match or directory.is_symlink() or not directory.is_dir():
            continue
        marker = directory / "checkpoint.json"
        if marker.is_symlink() or not marker.is_file():
            continue
        try:
            metadata = read_json(marker)
            progress = metadata["progress"]
            hashes = metadata["files_sha256"]
            step = int(match.group(1))
            if (
                metadata.get("schema") != 2
                or metadata.get("recipe_hash") != recipe_hash
                or progress["global_step"] != step
                or progress["next_window"] != step
                or not CHECKPOINT_REQUIRED_FILES <= hashes.keys()
                or not all(
                    portable_path(directory, name).is_file()
                    for name in CHECKPOINT_REQUIRED_FILES
                )
            ):
                continue
        except (OSError, ValueError, KeyError, TypeError, AttributeError):
            continue
        found.append((step, directory))
    return [directory for _, directory in sorted(found, key=lambda item: item[0])]
''')

replace_function(CORE, "prune_checkpoints", r'''
def prune_checkpoints(run, keep, recipe_hash=None):
    """Prune only recognized checkpoints in this session; leave other paths alone."""
    import shutil as _shutil
    if type(keep) is not int or keep < 1:
        raise ValueError("keep must be a positive integer")
    found = checkpoint_dirs(run, recipe_hash)
    removed = []
    for directory in found[:-keep]:
        _shutil.rmtree(directory)
        removed.append(str(directory))
    return removed, [str(directory) for directory in found[-keep:]]
''')

replace_function(CORE, "checkpoint_disk_plan", r'''
def checkpoint_disk_plan(
    trainable_numel, total_updates, updates_per_epoch, epochs,
    save_every_updates, keep, adapter_bytes=None, start_step=0
):
    """Estimate additional storage, including a checkpoint written before pruning.

    AdamW's two moment tensors are estimated at the parameter tensor byte size.
    Metadata/log/processor overhead is covered by an explicit safety margin.
    This is a free-space estimate, not a filesystem reservation.
    """
    if type(keep) is not int or keep < 1:
        raise ValueError("keep must be a positive integer")
    if not 0 <= start_step <= total_updates:
        raise ValueError("Invalid checkpoint starting step")
    if adapter_bytes is None:
        adapter_bytes = trainable_numel * 4
    if adapter_bytes <= 0:
        raise ValueError("Adapter size must be positive")
    schedule = [
        step for step in checkpoint_schedule(
            total_updates, updates_per_epoch, epochs, save_every_updates
        )
        if step > start_step
    ]
    per_checkpoint = adapter_bytes * 3
    retained = min(len(schedule), keep)
    in_flight = int(len(schedule) > retained)
    payload_peak = (
        per_checkpoint * (retained + in_flight)
        + adapter_bytes * 2
    )
    margin = max(2**30, math.ceil(payload_peak * 0.10))
    return {
        "trainable_numel": trainable_numel,
        "adapter_bytes": adapter_bytes,
        "bytes_per_checkpoint": per_checkpoint,
        "planned_checkpoints": len(schedule),
        "retained_checkpoints": retained,
        "start_step": start_step,
        "safety_margin_bytes": margin,
        "peak_run_bytes": payload_peak + margin,
        "estimate_basis": "parameter bytes plus two same-size AdamW moments",
    }
''')

replace(
    CORE,
    "def preprocessing_contract(c):",
    block(r'''
CHECKPOINT_REQUIRED_FILES = {
    "adapter_model.safetensors",
    "adapter_config.json",
    "stage_recipe.json",
    "training_state.pt",
}


def verify_training_checkpoint(folder, expected_recipe_hash=None):
    """Verify checkpoint files and recipe without unpickling optimizer state."""
    folder = Path(folder).resolve()
    metadata = read_json(folder / "checkpoint.json")
    if metadata.get("schema") != 2:
        raise ValueError(
            "Checkpoint predates the complete integrity manifest. "
            "Use its preserved original code to resume it."
        )
    hashes = metadata.get("files_sha256")
    if not isinstance(hashes, dict) or not CHECKPOINT_REQUIRED_FILES <= hashes.keys():
        raise ValueError("Incomplete checkpoint artifact manifest")
    for name, expected in hashes.items():
        path = portable_path(folder, name)
        if not path.is_file() or file_sha(path) != expected:
            raise ValueError(f"Checkpoint artifact missing/changed: {name}")
    recipe = read_json(folder / "stage_recipe.json")
    if digest(recipe) != metadata.get("recipe_hash"):
        raise ValueError("Checkpoint stage recipe integrity mismatch")
    if (
        expected_recipe_hash is not None
        and metadata["recipe_hash"] != expected_recipe_hash
    ):
        raise ValueError("Checkpoint belongs to a different recipe")
    progress = metadata.get("progress", {})
    step = progress.get("global_step")
    window = progress.get("next_window")
    if type(step) is not int or step < 1 or window != step:
        raise ValueError("Invalid checkpoint progress")
    match = re.fullmatch(r"step_(\d+)", folder.name)
    if match and int(match.group(1)) != step:
        raise ValueError("Checkpoint directory/progress mismatch")
    if not isinstance(metadata.get("trainable_parameters"), list):
        raise ValueError("Checkpoint parameter ordering is missing")
    return metadata, recipe
''') + "\ndef preprocessing_contract(c):",
)

# Bind the external parent recipe fields to the hashed saved recipe.
replace(
    CORE,
    '    if summary["recipe_hash"] != recipe["recipe_hash"]:\n',
    '    outer_recipe = {k: v for k, v in recipe.items()\n'
    '                    if k not in {"recipe_hash", "environment"}}\n'
    '    if digest(outer_recipe) != digest(saved_recipe):\n'
    '        raise ValueError("Parent recipe metadata changed")\n'
    '    if summary["recipe_hash"] != recipe["recipe_hash"]:\n',
)


# ---------------------------------------------------------------------------
# Export: one declared acceptance policy, complete diagnostics, no empty READY.
# ---------------------------------------------------------------------------

EXPORT = "12_export_sft_bundle.py"

replace(EXPORT, "import json\n", "import json\nimport math\n")

replace(
    EXPORT,
    '("zip", False), ("n", None)]',
    '("zip", False), ("n", None),\n'
    '                          ("acceptance_policy", "pixel-identical")]',
)

replace(
    EXPORT,
    "    started = time.perf_counter()\n",
    block(r'''
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
''').replace("\n", "\n    ").rstrip() + "\n"
    if False else
    '    if args.acceptance_policy not in {"pixel-identical", "validated"}:\n'
    '        raise ValueError("Unknown acceptance policy")\n'
    '    if args.n is not None and (type(args.n) is not int or args.n < 1):\n'
    '        raise ValueError("--n must be a positive integer")\n'
    '    if args.min_pages is not None and (\n'
    '        type(args.min_pages) is not int or args.min_pages < 1\n'
    '    ):\n'
    '        raise ValueError("--min-pages must be a positive integer")\n'
    '    if args.all and (args.n is not None or args.ids):\n'
    '        raise ValueError("--all cannot combine with --n or --ids")\n'
    '    started = time.perf_counter()\n',
)

replace(
    EXPORT,
    'str(resolve_source(root, x["html_path"]))',
    'str(resolve_source(root, x["html_path"]).resolve())',
)

replace(
    EXPORT,
    "    eligible = ok & set(man)\n",
    '''    missing_manifest = sorted(ok - set(man))
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
''',
)

replace(
    EXPORT,
    "    selected = set(ids)\n",
    '    selected = set(ids)\n'
    '    if not selected:\n'
    '        errors.append("No pages selected; an empty bundle is not allowed")\n',
)

replace(
    EXPORT,
    '''        if pid in selected:
            reason = "selected"
        elif r is None:
            reason = "no_final_validation_row"
        elif m is None and status == "ok":
            reason = "validated_but_absent_from_compressed_manifest"
        elif status != "ok":
            reason = f"final_validation_status={status}"
        elif pid in excluded_ids:
            reason = "reserved_by_exclude_ids"
        else:
            reason = "eligible_not_sampled"
''',
    '''        rejection = policy_reason(pid)
        if rejection is not None:
            reason = rejection
        elif pid in selected:
            reason = "selected"
        elif pid in excluded_ids:
            reason = "reserved_by_exclude_ids"
        else:
            reason = "eligible_not_sampled"
''',
)

replace(
    EXPORT,
    '    counts = {"final_validation_rows": len(final),',
    '    counts = {"acceptance_policy": args.acceptance_policy,\n'
    '              "final_validation_rows": len(final),',
)

replace(
    EXPORT,
    '''            if r.get("status") != "ok" or not truth(r.get("final_pixel_identical")):
                raise ValueError("requires Step 08 status=ok and final_pixel_identical=True")
''',
    '''            rejection = policy_reason(pid)
            if rejection is not None:
                raise ValueError(rejection)
''',
)

replace(
    EXPORT,
    '                    same = float(r[a]) == float(m[b]) and float(r[a]) >= 0\n',
    '                    left, right = float(r[a]), float(m[b])\n'
    '                    same = (math.isfinite(left) and left.is_integer()\n'
    '                            and left >= 0 and left == right)\n',
)

audit_block = '''    out.mkdir(parents=True)
    write_csv(out / "reconciliation.csv", reconciliation)
    write_csv(out / "excluded_pages.csv", [row for row in reconciliation if not row["selected"]])
    write_json(out / "reconciliation_summary.json", counts)
'''
replace(EXPORT, audit_block, "")
replace(
    EXPORT,
    "    if errors:\n        shown = errors[:40]\n",
    audit_block
    + '''    write_json(out / "export_issues.json", {
        "validation_passed": not errors,
        "errors": errors,
        "acceptance_policy": args.acceptance_policy,
    })
    (out / "page_ids.txt").write_text(
        "\\n".join(ids) + ("\\n" if ids else ""), encoding="utf-8"
    )
    if errors:
        shown = errors[:40]
''',
)
replace(
    EXPORT,
    '"Export validation failed; no bundle written:\\n- "',
    'f"Export validation failed; no READY bundle written. Audit: {out}\\n- "',
)
replace(
    EXPORT,
    '"selection_mode": "ids" if requested else ("all" if args.all else "sample"),',
    '"selection_mode": "ids" if requested is not None else ("all" if args.all else "sample"),\n'
    '               "acceptance_policy": args.acceptance_policy,',
)
replace(
    EXPORT,
    '    (out / "page_ids.txt").write_text("\\n".join(row["page_id"] for row in rows) + "\\n", encoding="utf-8")\n',
    "",
)
replace(
    EXPORT,
    '    p.add_argument("--source", default="webcode2m")\n',
    '    p.add_argument("--source", default="webcode2m")\n'
    '    p.add_argument("--acceptance-policy",\n'
    '                   choices=["pixel-identical", "validated"],\n'
    '                   default="pixel-identical",\n'
    '                   help="Strict pixel identity by default. validated accepts "\n'
    '                        "Step 08 status=ok without adding a pixel-identity rule; "\n'
    '                        "neither mode rerenders or changes the frozen gate.")\n',
)


# ---------------------------------------------------------------------------
# Preparation: every policy check precedes READY; verify destination bytes.
# ---------------------------------------------------------------------------

PREP = "13_prepare_sft.py"
replace(PREP, "import datetime as dt\n", "import datetime as dt\nimport math\n")
replace(
    PREP,
    "def prepare(args):\n    from PIL import Image\n",
    '''def prepare(args):
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
''',
)
replace(
    PREP,
    '                shutil.copyfile(portable_path(bundle, row[key]), dest / (key + (".png" if key == "image" else ".html")))\n',
    '''                copied = dest / (key + (".png" if key == "image" else ".html"))
                shutil.copyfile(portable_path(bundle, row[key]), copied)
                if file_sha(copied) != row[key + "_sha256"]:
                    raise ValueError(f"Copied {key} bytes differ from the frozen bundle")
''',
)
replace(
    PREP,
    '    write_json(out / "preparation_issues.json", {"errors": errors,',
    '''    retained_count = len({
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
                                                "errors": errors,''',
)
replace(
    PREP,
    '    assert all([r["page_id"] for r in records if r["arm"] == arm] == ids for arm in arms)\n',
    '''    if not all(
        [r["page_id"] for r in records if r["arm"] == arm] == ids
        for arm in arms
    ):
        raise ValueError("Selected arm page cohorts differ; no READY dataset written")
''',
)
replace(
    PREP,
    '                "overflow_policy": args.overflow_policy, "preprocessing": contract,\n',
    '                "overflow_policy": args.overflow_policy, "preprocessing": contract,\n'
    '                "expect_pages": args.expect_pages,\n'
    '                "max_overflow_pages": args.max_overflow_pages,\n',
)
replace(
    PREP,
    '''    if args.expect_pages is not None and len(ids) != args.expect_pages:
        raise ValueError(f"READY wrote {len(ids)} pages but --expect-pages {args.expect_pages} was declared; "
                         f"inspect {out}/preparation_issues.json. The dataset directory is left in place for review.")
''',
    "",
)


# ---------------------------------------------------------------------------
# Trainer: complete checkpoint manifest and final-save recovery.
# ---------------------------------------------------------------------------

TRAIN = "14_train_sft.py"

replace(
    TRAIN,
    "    args = ap.parse_args()\n",
    '    ap.add_argument("--remaining-arms", type=int, default=1,\n'
    '                    help="Launcher free-space budget for remaining sequential arms")\n'
    '    args = ap.parse_args()\n'
    '    if args.remaining_arms < 1:\n'
    '        ap.error("--remaining-arms must be positive")\n',
)

replace(
    TRAIN,
    '''            resume_recipe = read_json(Path(args.resume) / "stage_recipe.json")
            parent = resume_recipe.get("parent_run")
''',
    '''            cp = Path(args.resume).resolve()
            prior, resume_recipe = verify_training_checkpoint(cp)
            parent = resume_recipe.get("parent_run")
''',
)

replace(
    TRAIN,
    '''            cp = Path(args.resume).resolve()
            prior = read_json(cp / "checkpoint.json")
            if prior["recipe_hash"] != recipe_hash:
                raise ValueError("Resume recipe differs (data/model/config/code/environment packages); use original recipe")
            if file_sha(cp / "training_state.pt") != prior["state_sha256"]:
                raise ValueError("Checkpoint state hash mismatch")
            if file_sha(cp / "adapter_model.safetensors") != prior["adapter_sha256"]:
                raise ValueError("Checkpoint adapter hash mismatch")
''',
    '''            if prior["recipe_hash"] != recipe_hash:
                raise ValueError(
                    "Resume recipe differs: use the original data/model/config/code/packages"
                )
''',
)

replace(
    TRAIN,
    '        write_json(out / "trainable_parameters.json", {"target_modules": targets,\n',
    '''        trainable_specs = [
            {"name": name, "shape": list(parameter.shape), "dtype": str(parameter.dtype)}
            for name, parameter in trainable
        ]
        if args.resume and prior["trainable_parameters"] != trainable_specs:
            raise ValueError("Checkpoint trainable parameter structure/order differs")
        write_json(out / "trainable_parameters.json", {"target_modules": targets,
''',
)

replace(
    TRAIN,
    '''                                    c["save_every_updates"], keep)
''',
    '''                                    c["save_every_updates"], keep,
                                    adapter_bytes=sum(
                                        parameter.numel() * parameter.element_size()
                                        for _, parameter in trainable
                                    ),
                                    start_step=prior["progress"]["global_step"]
                                    if args.resume else 0)
''',
)
replace(
    TRAIN,
    '''            plan.update({"planned_checkpoints": 0, "retained_checkpoints": 0, "peak_run_bytes": plan["adapter_bytes"]*2})
''',
    '''            smoke_payload = plan["adapter_bytes"] * 2
            smoke_margin = max(2**30, math.ceil(smoke_payload * 0.10))
            plan.update({
                "planned_checkpoints": 0,
                "retained_checkpoints": 0,
                "safety_margin_bytes": smoke_margin,
                "peak_run_bytes": smoke_payload + smoke_margin,
            })
''',
)
replace(
    TRAIN,
    '        required = 0 if args.smoke else plan["peak_run_bytes"]\n',
    '        required = plan["peak_run_bytes"] * args.remaining_arms\n'
    '        plan["remaining_arms_budget"] = args.remaining_arms\n',
)

replace(
    TRAIN,
    '            optimizer.load_state_dict(saved["optimizer"])\n',
    '''            if saved.get("trainable_parameters") != trainable_specs:
                raise ValueError("Optimizer parameter ordering differs from the checkpoint")
            if digest(saved.get("progress")) != digest(prior["progress"]):
                raise ValueError("Checkpoint progress metadata/state mismatch")
            optimizer.load_state_dict(saved["optimizer"])
''',
)
replace(
    TRAIN,
    '''        if state["next_window"] >= len(windows):
            raise ValueError("Checkpoint already completed all updates; its adapter is the completed training state")
''',
    '''        if (
            type(state["next_window"]) is not int
            or state["global_step"] != state["next_window"]
            or not 0 <= state["next_window"] <= len(windows)
        ):
            raise ValueError("Checkpoint progress does not fit this stage")
        finalization_only = bool(
            args.resume and state["next_window"] == len(windows)
        )
        summary.update({
            "finalization_only": finalization_only,
            "progress": state,
            "session_examples": 0,
            "session_tokens": {key: 0 for key in TOKEN_KEYS},
            "training_peak_allocated_bytes": None,
            "training_peak_reserved_bytes": None,
        })
        if finalization_only:
            print("All updates are complete; recovering final adapter export only.", flush=True)
''',
)

replace(
    TRAIN,
    '            torch.save({"progress": state, "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),\n',
    '            torch.save({"progress": state, "trainable_parameters": trainable_specs,\n'
    '                        "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),\n',
)
replace(
    TRAIN,
    '''            write_json(temp / "checkpoint.json", {"recipe_hash": recipe_hash,
                       "state_sha256": file_sha(temp / "training_state.pt"),
                       "adapter_sha256": file_sha(temp / "adapter_model.safetensors"), "progress": state})
''',
    '''            artifact_hashes = {
                str(path.relative_to(temp)): file_sha(path)
                for path in temp.rglob("*") if path.is_file()
            }
            write_json(temp / "checkpoint.json", {
                "schema": 2,
                "recipe_hash": recipe_hash,
                "files_sha256": artifact_hashes,
                "state_sha256": artifact_hashes["training_state.pt"],
                "adapter_sha256": artifact_hashes["adapter_model.safetensors"],
                "trainable_parameters": trainable_specs,
                "progress": state,
            })
''',
)
replace(
    TRAIN,
    '            removed, retained = prune_checkpoints(out, keep)\n',
    '            removed, retained = prune_checkpoints(out, keep, recipe_hash)\n',
)


# ---------------------------------------------------------------------------
# Launcher: validate duplicate arguments before creating output and budget pair.
# ---------------------------------------------------------------------------

LAUNCH = "run_all_arms.py"

# The pasted launcher contains trailing spaces; normalize this modified file.
CHANGES[LAUNCH] = "\n".join(
    line.rstrip() for line in current(LAUNCH).splitlines()
) + "\n"

replace(
    LAUNCH,
    '''    root = Path(a.out).resolve()
    root.mkdir(parents=True, exist_ok=False)
    if len(set(a.arms)) != len(a.arms):
        p.error("Duplicate arms")
    for arm in a.arms:
''',
    '''    if len(set(a.arms)) != len(a.arms):
        p.error("Duplicate arms")
    root = Path(a.out).resolve()
    root.mkdir(parents=True, exist_ok=False)
    for arm_index, arm in enumerate(a.arms):
''',
)
replace(
    LAUNCH,
    '                   "--arm", arm, "--out", str(root / arm)]\n',
    '                   "--arm", arm, "--out", str(root / arm),\n'
    '                   "--remaining-arms", str(len(a.arms) - arm_index)]\n',
)


# ---------------------------------------------------------------------------
# Verification: retain final adapter checks; add checkpoint file verification.
# ---------------------------------------------------------------------------

CHANGES["16_verify_sft_checkpoint.py"] = block(r'''
"""Check final adapter tensors or an intermediate checkpoint's artifact manifest."""
import argparse
from pathlib import Path
from sft_core import *


def check_tensors(folder, require_learned):
    import torch
    from safetensors import safe_open
    count, nonzero_b = 0, False
    with safe_open(
        folder / "adapter_model.safetensors", framework="pt", device="cpu"
    ) as handle:
        for key in handle.keys():
            value = handle.get_tensor(key)
            if not torch.isfinite(value).all().item():
                raise ValueError(f"Nonfinite tensor: {key}")
            count += value.numel()
            if "lora_B" in key and torch.count_nonzero(value).item():
                nonzero_b = True
    if not count:
        raise ValueError("Adapter contains no parameters")
    if require_learned and not nonzero_b:
        raise ValueError("All LoRA B matrices are zero")
    return count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--run")
    selection.add_argument("--checkpoint")
    args = parser.parse_args()

    if args.checkpoint:
        folder = Path(args.checkpoint)
        metadata, _ = verify_training_checkpoint(folder)
        count = check_tensors(folder, require_learned=False)
        print(
            f"PASS: step {metadata['progress']['global_step']}; "
            f"all checkpoint artifact hashes and {count:,} finite adapter parameters. "
            "Optimizer state was not deserialized; actual resume still needs a GPU check."
        )
        return

    root = Path(args.run)
    summary = read_json(root / "summary.json")
    if summary.get("status") != "completed":
        raise ValueError("Training/finalization not completed")
    folder = root / "final_adapter"
    hashes = read_json(root / "final_adapter_hashes.json")
    required = {
        "adapter_model.safetensors", "adapter_config.json",
        "training_recipe.json", "preprocessor_config.json",
    }
    if not required <= hashes.keys():
        raise ValueError("Incomplete final adapter manifest")
    for name, expected in hashes.items():
        path = portable_path(folder, name)
        if not path.is_file() or file_sha(path) != expected:
            raise ValueError(f"Changed/missing adapter artifact: {name}")
    recipe = read_json(root / "recipe.json")
    saved_recipe = read_json(folder / "training_recipe.json")
    outer = {
        key: value for key, value in recipe.items()
        if key not in {"recipe_hash", "environment"}
    }
    if (
        digest(saved_recipe) != recipe["recipe_hash"]
        or digest(outer) != recipe["recipe_hash"]
        or summary["recipe_hash"] != recipe["recipe_hash"]
    ):
        raise ValueError("Final recipe integrity mismatch")
    if (
        not summary.get("initial_adapter_digest")
        or not summary.get("final_adapter_digest")
        or summary["final_adapter_digest"] == summary["initial_adapter_digest"]
    ):
        raise ValueError("Missing/unchanged adapter digest")
    count = check_tensors(folder, require_learned=True)
    print(f"PASS: {count:,} finite learned adapter parameters and artifact hashes. No inference.")


if __name__ == "__main__":
    main()
''')


# ---------------------------------------------------------------------------
# Configurations: identical learning/preprocessing settings across both stages.
# ---------------------------------------------------------------------------

import json

BASE_CONFIG = {
    "model_id": "Qwen/Qwen2.5-VL-7B-Instruct",
    "epochs": 2,
    "seed": 42,
    "gradient_accumulation_steps": 8,
    "learning_rate": 0.0001,
    "weight_decay": 0.01,
    "warmup_ratio": 0.1,
    "loss_normalization": "example",
    "lora_r": 64,
    "lora_alpha": 128,
    "lora_dropout": 0.0,
    "max_seq_length": 12288,
    "min_image_tokens": 256,
    "max_image_tokens": 1280,
    "warmup_metric_updates": 5,
    "max_grad_norm": 1.0,
    "keep_last_checkpoints": 3,
}
for filename, experiment, interval in [
    ("stage1_2250_12k.json", "stage1_2250_two_arms_12k", 50),
    ("stage2_7500_12k.json", "stage2_7500_two_arms_12k", 100),
]:
    config = BASE_CONFIG | {
        "experiment": experiment,
        "save_every_updates": interval,
    }
    CHANGES["configs/" + filename] = json.dumps(config, indent=2) + "\n"


# Export failures now deliberately leave an audit, but never a READY bundle.
replace(
    "tests/test_core_and_export.py",
    "self.assertFalse(Path(args.out).exists())",
    'self.assertFalse((Path(args.out) / "bundle.json").exists())',
)


# ---------------------------------------------------------------------------
# Focused CPU regression tests. No GPU/model downloads and no synthetic results
# are represented as research measurements.
# ---------------------------------------------------------------------------

CHANGES["tests/test_stage_ready_patch.py"] = block(r'''
import argparse
import contextlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import sft_core as core


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


exporter = module("patched_exporter", ROOT / "12_export_sft_bundle.py")
preparer = module("patched_preparer", ROOT / "13_prepare_sft.py")
historical = module("historical_export_tests", ROOT / "tests/test_core_and_export.py")


class Checkpoints(unittest.TestCase):
    def fixture(self, root, step):
        recipe = {"fixture": "same-stage"}
        recipe_hash = core.digest(recipe)
        core.write_json(root / "recipe.json", {"recipe_hash": recipe_hash})
        folder = root / "checkpoints" / f"step_{step:06d}"
        folder.mkdir(parents=True)
        core.write_json(folder / "stage_recipe.json", recipe)
        core.write_json(folder / "adapter_config.json", {"r": 64})
        (folder / "adapter_model.safetensors").write_bytes(b"fixture")
        (folder / "training_state.pt").write_bytes(b"fixture")
        hashes = {p.name: core.file_sha(p) for p in folder.iterdir()}
        core.write_json(folder / "checkpoint.json", {
            "schema": 2, "recipe_hash": recipe_hash,
            "files_sha256": hashes,
            "trainable_parameters": [],
            "progress": {"global_step": step, "next_window": step},
        })
        return folder, recipe_hash

    def test_retention_ignores_partial_foreign_and_symlink(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for step in (1, 2, 10):
                latest, recipe_hash = self.fixture(root, step)
            parent = root / "checkpoints"
            for name in ("step_999999.incomplete", "foreign"):
                folder = parent / name
                folder.mkdir()
                core.write_json(folder / "checkpoint.json", {})
            (parent / "step_000011").symlink_to(latest, target_is_directory=True)
            removed, retained = core.prune_checkpoints(root, 2, recipe_hash)
            self.assertEqual(len(removed), 1)
            self.assertEqual([Path(x).name for x in retained],
                             ["step_000002", "step_000010"])
            self.assertTrue((parent / "step_999999.incomplete").exists())
            self.assertTrue((parent / "foreign").exists())
            self.assertTrue((parent / "step_000011").is_symlink())

    def test_config_corruption_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            folder, recipe_hash = self.fixture(Path(directory), 3)
            core.verify_training_checkpoint(folder, recipe_hash)
            (folder / "adapter_config.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "adapter_config"):
                core.verify_training_checkpoint(folder, recipe_hash)

    def test_schedule_and_finalization_disk_budget(self):
        self.assertEqual(len(core.checkpoint_schedule(560, 280, 2, 50)), 13)
        plan = core.checkpoint_disk_plan(100, 560, 280, 2, 50, 3)
        self.assertEqual(plan["retained_checkpoints"], 3)
        self.assertGreaterEqual(plan["peak_run_bytes"], 4 * 1200 + 800)
        final = core.checkpoint_disk_plan(
            100, 560, 280, 2, 50, 3, start_step=560
        )
        self.assertEqual(final["planned_checkpoints"], 0)
        self.assertGreater(final["peak_run_bytes"], 0)


class ExportPolicies(unittest.TestCase):
    def fixture(self, root):
        args = historical.TestExport().fixture(root)
        args.n = None
        setattr(args, "all", True)
        args.zip = False
        return args

    def test_missing_validated_manifest_row_fails_with_audit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = self.fixture(root)
            manifest = root / "data/splits/compressed_webcode2m_manifest.csv"
            core.write_csv(manifest, core.read_csv(manifest)[1:])
            with self.assertRaisesRegex(ValueError, "no compressed-manifest"):
                exporter.export(args)
            self.assertTrue((Path(args.out) / "export_issues.json").is_file())
            self.assertFalse((Path(args.out) / "bundle.json").exists())

    def test_strict_and_validated_policies_are_explicit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = self.fixture(root)
            report = root / "reports/csv/final_validation_webcode2m.csv"
            rows = core.read_csv(report)
            rows[0]["final_pixel_identical"] = False
            core.write_csv(report, rows)
            args.plan_only = True
            exporter.export(args)
            audit = core.read_json(Path(args.out) / "reconciliation_summary.json")
            self.assertEqual(audit["selected_pages"], 2)
            self.assertTrue((Path(args.out) / "page_ids.txt").is_file())
            args.out = str(root / "validated_plan")
            args.acceptance_policy = "validated"
            exporter.export(args)
            audit = core.read_json(Path(args.out) / "reconciliation_summary.json")
            self.assertEqual(audit["selected_pages"], 3)

    def test_empty_id_selection_never_becomes_ready(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = self.fixture(root)
            setattr(args, "all", False)
            path = root / "empty.txt"
            path.write_text("")
            args.ids = str(path)
            with self.assertRaisesRegex(ValueError, "No pages selected"):
                exporter.export(args)
            self.assertFalse((Path(args.out) / "bundle.json").exists())

    def test_equivalent_original_paths_are_deduplicated(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            row = {"ladder": json.dumps([
                {"level": "original", "html_path": "inputs/a.html"},
                {"level": "original", "html_path": "inputs/../inputs/a.html"},
            ])}
            self.assertEqual(
                exporter.original_from_ladder(root, row),
                (root / "inputs/a.html").resolve(),
            )


class Vector(list):
    def tolist(self):
        return list(self)


class Preparation(unittest.TestCase):
    def run_fixture(self, root, expect, long_page=False, budget=None, corrupt=False):
        bundle = root / "bundle"
        bundle.mkdir()
        rows = []
        for index in range(2):
            folder = bundle / str(index)
            folder.mkdir()
            row = {"page_id": str(index), "final_level": "l1"}
            for key in ("original", "verified", "image"):
                path = folder / key
                payload = (
                    b"LONG" if long_page and index == 0 and key == "original"
                    else b"short"
                )
                path.write_bytes(payload)
                row[key] = str(path.relative_to(bundle))
                row[key + "_sha256"] = core.file_sha(path)
            rows.append(row)
        core.write_jsonl(bundle / "pages.jsonl", rows)
        core.write_json(bundle / "bundle.json", {
            "ready": True, "pages": 2,
            "pages_sha256": core.file_sha(bundle / "pages.jsonl"),
        })
        config = core.read_json(ROOT / "configs/stage1_2250_12k.json")
        config["max_seq_length"] = 8
        core.write_json(root / "config.json", config)
        core.write_json(root / "lock.json", {
            "model_id": core.MODEL_ID, "revision": "a" * 40,
        })
        args = argparse.Namespace(
            bundle=str(bundle), config=str(root / "config.json"),
            model_lock=str(root / "lock.json"), out=str(root / "prepared"),
            arms=["original", "verified"], overflow_policy="common-fit",
            expect_pages=expect, max_overflow_pages=budget,
        )
        tokenizer = types.SimpleNamespace(
            all_special_tokens=[],
            convert_tokens_to_ids=lambda token: 99,
            encode=lambda text, **kwargs: list(text),
        )
        processor = types.SimpleNamespace(
            tokenizer=tokenizer,
            save_pretrained=lambda path: Path(path).mkdir(),
        )
        image = types.SimpleNamespace(close=lambda: None)
        opened = types.SimpleNamespace(convert=lambda mode: image)
        fake_pil = types.ModuleType("PIL")
        fake_pil.Image = types.SimpleNamespace(
            open=lambda path: contextlib.nullcontext(opened)
        )

        def encode(processor, image, text):
            ids = [9, 99]
            if text is not None:
                ids += list(range(10, 18)) if text == "LONG" else [10, 11]
            return {
                "input_ids": [Vector(ids)],
                "image_grid_thw": Vector([[1, 2, 2]]),
            }

        real_copy = preparer.shutil.copyfile

        def copy(source, destination):
            result = real_copy(source, destination)
            if corrupt and Path(destination).name == "original.html":
                Path(destination).write_bytes(b"changed")
            return result

        with contextlib.ExitStack() as stack:
            stack.enter_context(mock.patch.dict(sys.modules, {"PIL": fake_pil}))
            overrides = {
                "verify_model_lock": lambda lock: root,
                "load_processor": lambda *args: processor,
                "conversation_text": lambda processor, html=None: html,
                "encode_image_text": encode,
                "package_versions": lambda: {"fixture": "1"},
            }
            for name, value in overrides.items():
                stack.enter_context(mock.patch.object(preparer, name, value))
            stack.enter_context(mock.patch.object(preparer.shutil, "copyfile", copy))
            preparer.prepare(args)
        return Path(args.out)

    def test_count_mismatch_leaves_no_ready_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(ValueError, "Preflight failed"):
                self.run_fixture(root, expect=3)
            self.assertFalse((root / "prepared/prepared.json").exists())
            issues = core.read_json(root / "prepared/preparation_issues.json")
            self.assertEqual(issues["candidate_retained_pages"], 2)

    def test_common_fit_excludes_one_page_from_both_arms(self):
        with tempfile.TemporaryDirectory() as directory:
            out = self.run_fixture(Path(directory), expect=1, long_page=True, budget=1)
            for arm in ("original", "verified"):
                rows = core.read_jsonl(out / f"{arm}.jsonl")
                self.assertEqual([row["page_id"] for row in rows], ["1"])
            self.assertTrue(core.read_json(out / "prepared.json")["ready"])

    def test_overflow_budget_leaves_no_ready_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(ValueError, "declared budget"):
                self.run_fixture(root, expect=1, long_page=True, budget=0)
            self.assertFalse((root / "prepared/prepared.json").exists())

    def test_changed_copy_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(ValueError, "Preflight failed"):
                self.run_fixture(root, expect=2, corrupt=True)
            self.assertFalse((root / "prepared/prepared.json").exists())


if __name__ == "__main__":
    unittest.main()
''')


# ---------------------------------------------------------------------------
# Validate the staged code, then back up and apply. No GPU work occurs here.
# ---------------------------------------------------------------------------

for name, text in CHANGES.items():
    if name.endswith(".py"):
        compile(text, str(ROOT / name), "exec")
    elif name.endswith(".json"):
        json.loads(text)

stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
output_parent = ROOT.parent

with tempfile.TemporaryDirectory(
    prefix="sft_patch_test_", dir=str(output_parent)
) as temporary:
    staged = Path(temporary) / ROOT.name
    shutil.copytree(
        ROOT, staged,
        ignore=shutil.ignore_patterns(
            "__pycache__", ".git", ".venv", "venv", "*.pyc", "*.zip",
            "*.tar.gz", "outputs", "reports",
        ),
    )
    for name, text in CHANGES.items():
        path = staged / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8", newline="\n")

    for pattern in ("test_stage_ready_patch.py", "test_core_and_export.py"):
        subprocess.run(
            [
                sys.executable, "-m", "unittest", "discover",
                "-s", "tests", "-p", pattern, "-v",
            ],
            cwd=staged, check=True,
        )

backup = output_parent / f"sft_before_ready_patch_{stamp}.tar.gz"
with tarfile.open(backup, "x:gz") as archive:
    for name in sorted(set(CHANGES) | {"SHA256SUMS.txt"}):
        path = ROOT / name
        if path.is_file():
            archive.add(path, arcname=f"{ROOT.name}/{name}")

original_bytes = {
    name: (ROOT / name).read_bytes() if (ROOT / name).is_file() else None
    for name in set(CHANGES) | {"SHA256SUMS.txt"}
}
try:
    for name, text in CHANGES.items():
        path = ROOT / name
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".patch-tmp")
        temporary.write_text(text, encoding="utf-8", newline="\n")
        os.replace(temporary, path)

    allowed_suffixes = {".py", ".sh", ".json", ".md", ".txt", ".toml", ".yaml", ".yml"}

    def source_files():
        return sorted(
            path for path in ROOT.rglob("*")
            if path.is_file()
            and not path.is_symlink()
            and path.suffix in allowed_suffixes
            and not any(
                part.startswith(".") or part in {"__pycache__", "venv", "outputs", "reports"}
                for part in path.relative_to(ROOT).parts
            )
        )

    lines = []
    for path in source_files():
        if path.name == "SHA256SUMS.txt":
            continue
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        lines.append(f"{sha}  {path.relative_to(ROOT).as_posix()}")
    (ROOT / "SHA256SUMS.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
except BaseException:
    for name, payload in original_bytes.items():
        path = ROOT / name
        if payload is None:
            path.unlink(missing_ok=True)
        else:
            path.write_bytes(payload)
    raise

archive_path = output_parent / f"qwen_sft_ready_{stamp}.zip"
with zipfile.ZipFile(
    archive_path, "x", compression=zipfile.ZIP_DEFLATED
) as archive:
    for path in source_files():
        archive.write(path, f"{ROOT.name}/{path.relative_to(ROOT).as_posix()}")
with zipfile.ZipFile(archive_path) as archive:
    bad = archive.testzip()
    if bad:
        raise RuntimeError(f"Archive integrity check failed: {bad}")

print("\nPatch applied after staged CPU tests passed.")
print("Backup:", backup)
print("Patched source/config/documentation ZIP:", archive_path)
print("Changed files:")
for name in sorted(CHANGES):
    print(" ", name)
print("  SHA256SUMS.txt")
print("GPU training, real-data compatibility and actual resume remain to be checked.")