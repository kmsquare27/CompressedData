"""Check saved adapter integrity and finite learned weights; does not run inference."""
import argparse
from pathlib import Path
from sft_core import *

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", required=True)
    args = p.parse_args()
    root = Path(args.run)
    s = read_json(root / "summary.json")
    assert s["status"] == "completed", "Training not completed"
    folder = root / "final_adapter"
    hashes = read_json(root / "final_adapter_hashes.json")
    for name, expected in hashes.items():
        assert file_sha(portable_path(folder, name)) == expected, f"Changed adapter artifact: {name}"
    assert s["final_adapter_digest"] != s["initial_adapter_digest"], "Adapter unchanged"
    import torch
    from safetensors import safe_open
    count, nonzero_b = 0, False
    with safe_open(folder / "adapter_model.safetensors", framework="pt", device="cpu") as f:
        for key in f.keys():
            value = f.get_tensor(key)
            assert torch.isfinite(value).all(), f"Nonfinite tensor: {key}"
            count += value.numel()
            if "lora_B" in key and torch.count_nonzero(value).item():
                nonzero_b = True
    assert nonzero_b, "All LoRA B matrices are zero"
    assert (folder / "adapter_config.json").is_file()
    assert (folder / "preprocessor_config.json").is_file()
    print(f"PASS: {count:,} saved finite adapter parameters; learned LoRA B weights found. No inference performed.")

if __name__ == "__main__":
    main()
