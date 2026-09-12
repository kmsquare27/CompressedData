"""Run on RunPod before downloading/training. Exercises actual CUDA and LoRA backward."""
import argparse
import platform
import subprocess
import sys
from pathlib import Path
from sft_core import package_versions, write_json

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    import torch
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    from peft import LoraConfig, get_peft_model
    assert sys.version_info[:2] in {(3, 11), (3, 12)}, "Use Python 3.11 or 3.12"
    assert torch.cuda.is_available(), "CUDA unavailable; check NVIDIA driver and PyTorch CUDA wheel"
    assert torch.cuda.device_count() == 1, "Set CUDA_VISIBLE_DEVICES to exactly one GPU"
    assert torch.cuda.is_bf16_supported(), "This recipe requires a BF16-capable GPU"
    x = torch.randn(2, 32, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    base = torch.nn.Sequential(torch.nn.Linear(32, 16, bias=False)).to(device="cuda", dtype=torch.bfloat16)
    layer = get_peft_model(base, LoraConfig(r=64, lora_alpha=128, target_modules=["0"], bias="none"))
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss = layer(x).float().square().mean()
    loss.backward()
    learned = [p for n, p in layer.named_parameters() if "lora_B" in n]
    assert learned and all(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0 for p in learned), "LoRA backward failed"
    props = torch.cuda.get_device_properties(0)
    result = {"python": sys.version, "platform": platform.platform(), "packages": package_versions(),
              "gpu": props.name, "vram_gib": props.total_memory/2**30,
              "compute_capability": [props.major, props.minor], "cuda_runtime": torch.version.cuda,
              "lora_bf16_backward": "passed"}
    try:
        result["nvidia_smi"] = subprocess.check_output(["nvidia-smi"], text=True)
    except Exception as e:
        result["nvidia_smi_error"] = str(e)
    write_json(args.out, result)
    print(result["gpu"], round(result["vram_gib"], 1), "GiB; LoRA/BF16 backward passed")
    if result["vram_gib"] < 23:
        print("Memory may be insufficient. The full-model smoke run, not this tiny check, determines feasibility.")

if __name__ == "__main__":
    main()
