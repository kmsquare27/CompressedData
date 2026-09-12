"""Run paired arms sequentially on the SAME GPU. Each arm initializes from the locked base."""
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
    p.add_argument("--init-from-runs", help="Previous run root with original/, naive/, verified/ completed runs")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS), help="For repeated trials, vary order to reduce thermal/cache order effects")
    a = p.parse_args()
    if a.smoke and a.init_from_runs:
        p.error("Continuation cannot use --smoke")
    root = Path(a.out).resolve()
    root.mkdir(parents=True, exist_ok=False)
    if len(set(a.arms)) != len(a.arms):
        p.error("Duplicate arms")
    for arm in a.arms:
        command = [sys.executable, str(Path(__file__).with_name("14_train_sft.py")),
                   "--prepared", a.prepared, "--config", a.config, "--model-lock", a.model_lock,
                   "--arm", arm, "--out", str(root / arm)]
        if a.init_from_runs:
            command.extend(["--init-from-run", str(Path(a.init_from_runs).resolve() / arm)])
        if a.smoke:
            command.append("--smoke")
        print("Starting", arm, flush=True)
        subprocess.run(command, check=True)

if __name__ == "__main__":
    main()
