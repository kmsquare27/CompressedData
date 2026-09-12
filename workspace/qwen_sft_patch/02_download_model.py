"""Resolve main ONCE to a commit; reuse this immutable snapshot for all arms."""
import argparse
from pathlib import Path
from sft_core import MODEL_ID, file_sha, write_json, read_json, verify_model_lock
import os

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--revision", default="main")
    p.add_argument("--out", required=True)
    args = p.parse_args()
    os.environ.setdefault("HF_HOME", "/workspace/hf-cache")
    if Path(args.out).exists():
        lock = read_json(args.out)
        if args.revision != "main" and args.revision != lock["revision"]:
            p.error("Existing lock has another revision; use a new lock path")
        print("Reusing locked model:", verify_model_lock(lock))
        return
    from huggingface_hub import HfApi, snapshot_download
    commit = HfApi().model_info(MODEL_ID, revision=args.revision).sha
    directory = Path(snapshot_download(MODEL_ID, revision=commit))
    hashes = {str(f.relative_to(directory)): file_sha(f) for f in directory.rglob("*") if f.is_file()}
    write_json(args.out, {"model_id": MODEL_ID, "revision": commit,
                         "snapshot_path": str(directory.resolve()), "files_sha256": hashes})
    print(f"Locked {MODEL_ID} @ {commit}; downloaded to {directory}")

if __name__ == "__main__":
    main()
