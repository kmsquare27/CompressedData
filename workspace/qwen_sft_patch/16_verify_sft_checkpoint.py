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
