"""Run original and verified by default, sequentially on the SAME GPU."""
import argparse
import subprocess
import sys
from pathlib import Path
from sft_core import ARMS

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prepared", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--model-lock", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--init-from-runs", help="Previous run root with a completed same-arm run for each selected arm")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--keep-all-checkpoints", action="store_true", help="Pass through to the trainer; retains every checkpoint")
    p.add_argument("--allow-low-disk", action="store_true", help="Pass through to the trainer's free-space preflight")
    p.add_argument("--arms", nargs="+", choices=ARMS, default=["original", "verified"], help="Default: original verified. Include naive explicitly to train it; list order is execution order.")
    a = p.parse_args()
    if len(set(a.arms)) != len(a.arms):
        p.error("Duplicate arms")
    root = Path(a.out).resolve()
    root.mkdir(parents=True, exist_ok=False)
    for arm_index, arm in enumerate(a.arms):
        command = [sys.executable, str(Path(__file__).with_name("14_train_sft.py")),
                   "--prepared", a.prepared, "--config", a.config, "--model-lock", a.model_lock,
                   "--arm", arm, "--out", str(root / arm),
                   "--remaining-arms", str(len(a.arms) - arm_index)]
        if a.init_from_runs:
            command.extend(["--init-from-run", str(Path(a.init_from_runs).resolve() / arm)])
        if a.smoke:
            command.append("--smoke")
        if a.keep_all_checkpoints:
            command.append("--keep-all-checkpoints")
        if a.allow_low_disk:
            command.append("--allow-low-disk")
        print("Starting", arm, flush=True)
        subprocess.run(command, check=True)

if __name__ == "__main__":
    main()
