"""Record real subprocess elapsed time, then report preprocessing break-even.

Standard library only. Install in qwen_sft_patch/compute_tools. Commands are
executed as argument lists (no shell). A record is one sequential stage, NOT
per-page timing. Use StageTimer inside a page loop for optional page detail.
"""
from __future__ import annotations
import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import platform
import subprocess
import sys
import time
import uuid
from pathlib import Path


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def append(path, row):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, allow_nan=False) + "\n")


class StageTimer:
    """Optional page detail. Use a separate ledger for each parallel worker.

    with StageTimer('pages.jsonl', run_id='batch1', stage='l1', page_id=pid):
        process_page(pid)
    Page records are never summed into the sequential-stage elapsed total.
    """
    def __init__(self, ledger, *, run_id, stage, page_id):
        self.ledger = ledger
        self.row = dict(kind="page", run_id=run_id, stage=stage, page_id=str(page_id))

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        append(self.ledger, self.row | dict(elapsed_seconds=time.perf_counter()-self.start,
               status="failed" if exc_type else "completed"))
        return False


def record(a):
    command = a.command[1:] if a.command[:1] == ["--"] else a.command
    if not command:
        raise ValueError("Supply a command after --")
    ids = [x.strip() for x in Path(a.page_ids).read_text(encoding="utf-8-sig").splitlines() if x.strip()]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("Page-ID file must contain unique IDs, one per line")
    identity = hashlib.sha256(json.dumps(sorted(ids)).encode()).hexdigest()
    row = dict(schema=1, kind="stage", record_id=str(uuid.uuid4()), run_id=a.run_id,
               stage=a.stage, variant=a.variant, hardware_label=a.hardware_label,
               platform=platform.platform(), pages=len(ids), page_ids_sha256=identity,
               command=command, cwd=str(Path(a.cwd).resolve()),
               started_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
               hourly_rate=a.hourly_rate, currency=a.currency, status="failed")
    start = time.perf_counter()
    try:
        result = subprocess.run(command, cwd=a.cwd, check=False)
        row.update(returncode=result.returncode,
                   status="completed" if result.returncode == 0 else "failed")
    except BaseException as exc:
        row["error"] = repr(exc)
        raise
    finally:
        row["elapsed_seconds"] = time.perf_counter()-start
        append(a.ledger, row)
    return result.returncode


def stage_totals(rows, run_id, variant, stages):
    chosen = [r for r in rows if r.get("kind") == "stage" and r.get("run_id") == run_id
              and r.get("variant") == variant]
    if not chosen:
        raise ValueError(f"No stage records for {run_id}/{variant}")
    if any(r.get("status") != "completed" for r in chosen):
        raise ValueError("Failed attempts present: retain them, use a fresh run ID for canonical timing")
    names = [r["stage"] for r in chosen]
    if len(names) != len(set(names)) or set(names) != set(stages):
        raise ValueError("Exactly one successful record per declared stage is required")
    if len({(r["page_ids_sha256"], r["pages"]) for r in chosen}) != 1:
        raise ValueError("Stage input page cohorts differ")
    for r in chosen:
        if not math.isfinite(r["elapsed_seconds"]) or r["elapsed_seconds"] < 0:
            raise ValueError("Invalid elapsed time")
    # Overlapping wrapper intervals would double count elapsed time.
    intervals = sorted((dt.datetime.fromisoformat(r["started_utc"]).timestamp(), r["elapsed_seconds"]) for r in chosen)
    if any(t+s > u+0.05 for (t,s),(u,_) in zip(intervals, intervals[1:])):
        raise ValueError("Stages overlap; sequential elapsed-time accounting is not valid")
    rates = [r.get("hourly_rate") for r in chosen]
    currencies = {r.get("currency") for r in chosen if r.get("hourly_rate") is not None}
    if len(currencies) > 1:
        raise ValueError("Mixed currencies")
    return dict(seconds=sum(r["elapsed_seconds"] for r in chosen),
                cost=sum(r["elapsed_seconds"]*r["hourly_rate"]/3600 for r in chosen) if all(v is not None for v in rates) else None,
                currency=next(iter(currencies), None), pages=chosen[0]["pages"],
                page_ids_sha256=chosen[0]["page_ids_sha256"], stages=names)


def break_even(extra, saved):
    if saved <= 0:
        return dict(epochs=None, reason="no_positive_training_saving")
    return dict(epochs=max(0.0, extra)/saved, reason="already_repaid" if extra <= 0 else "linear_projection")


def paired_training(original, verified):
    for s, arm in [(original, "original"), (verified, "verified")]:
        if s.get("status") != "completed" or s.get("smoke") or s.get("resumed") or s.get("arm") != arm:
            raise ValueError("Use completed, non-smoke, uninterrupted original/verified training summaries")
    keys = ["config", "data_fingerprint", "model_revision", "code_hashes", "parent_comparison_key"]
    if any(original.get(k) != verified.get(k) for k in keys):
        raise ValueError("Training configurations/data/model/code/predecessor do not match")
    for s in [original, verified]:
        if not s.get("environment") or not s.get("pages") or s["config"].get("epochs", 0) <= 0:
            raise ValueError("Training metadata incomplete")
        for key in ["training_loop_excluding_checkpoint_seconds", "job_wall_seconds"]:
            value = s.get(key)
            if not isinstance(value,(int,float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"Missing/invalid training measurement: {key}")
    env = lambda s: {k:v for k,v in s["environment"].items() if k not in {"gpu_uuid", "platform"}}
    if env(original) != env(verified) or original["pages"] != verified["pages"]:
        raise ValueError("Training environment/page count mismatch")


def report(a):
    rows = [json.loads(line) for line in Path(a.ledger).read_text().splitlines() if line.strip()]
    v = stage_totals(rows, a.run_id, "verified", a.verified_stages)
    b = stage_totals(rows, a.run_id, "original", a.original_stages)
    if (v["pages"], v["page_ids_sha256"]) != (b["pages"], b["page_ids_sha256"]):
        raise ValueError("Baseline and verified preprocessing cohorts differ")
    orig, ver = read_json(a.original_summary), read_json(a.verified_summary)
    paired_training(orig, ver)
    # Require Step 15's reconciliation result rather than treating any summary as valid.
    with Path(a.compute_summary).open(encoding="utf-8-sig", newline="") as f:
        checked = list(csv.DictReader(f))
    for path in [a.original_summary, a.verified_summary]:
        expected = Path(path).resolve().parent
        matches = [r for r in checked if Path(r["session"]).resolve() == expected]
        if len(matches) != 1 or matches[0]["eligible_for_paired_comparison"].lower() != "true":
            raise ValueError("Run is absent/ineligible in Step 15 compute_summary.csv; report on the same machine")
    metric = "training_loop_excluding_checkpoint_seconds"
    epoch_saving = (orig[metric]-ver[metric])/orig["config"]["epochs"]
    extra = v["seconds"]-b["seconds"]
    out = dict(schema=1, original_preprocessing=b, verified_preprocessing=v,
               preprocessing_pages=v["pages"], training_pages=orig["pages"],
               scope_note=a.scope_note, incremental_preprocessing_seconds=extra,
               training_seconds_saved_per_epoch=epoch_saving,
               elapsed_break_even=break_even(extra, epoch_saving),
               assumptions="Measured sequential preprocessing service time; excludes idle gaps. Linear epoch projection excludes training model-load and checkpoint costs. Not CPU core-seconds or FLOPs.",
               training_job_seconds_saved=orig["job_wall_seconds"]-ver["job_wall_seconds"])
    if a.gpu_hourly_rate is not None and b["cost"] is not None and v["cost"] is not None:
        if b["currency"] != v["currency"] or b["currency"] != a.currency:
            raise ValueError("Preprocessing and GPU cost currencies differ")
        out.update(currency=a.currency, incremental_preprocessing_cost=v["cost"]-b["cost"],
                   money_break_even=break_even(v["cost"]-b["cost"], epoch_saving*a.gpu_hourly_rate/3600))
    target = Path(a.out)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8") as f:
        json.dump(out, f, indent=2, allow_nan=False)
    print(json.dumps(out, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="action", required=True)
    r = sub.add_parser("record", help="Wrap one real pipeline command")
    for k in ["ledger", "run-id", "stage", "page-ids", "hardware-label"]:
        r.add_argument("--"+k, required=True)
    r.add_argument("--variant", choices=["original", "verified"], required=True)
    r.add_argument("--cwd", default=".")
    r.add_argument("--hourly-rate", type=float)
    r.add_argument("--currency", default="USD")
    r.add_argument("command", nargs=argparse.REMAINDER)
    q = sub.add_parser("report")
    for k in ["ledger", "run-id", "original-summary", "verified-summary", "compute-summary", "scope-note", "out"]:
        q.add_argument("--"+k, required=True)
    q.add_argument("--original-stages", nargs="+", required=True)
    q.add_argument("--verified-stages", nargs="+", required=True)
    q.add_argument("--gpu-hourly-rate", type=float)
    q.add_argument("--currency", default="USD")
    a = p.parse_args()
    for k in ["hourly_rate", "gpu_hourly_rate"]:
        v = getattr(a, k, None)
        if v is not None and (not math.isfinite(v) or v < 0):
            p.error("Rates must be finite and nonnegative")
    if a.action == "record":
        sys.exit(record(a))
    report(a)


if __name__ == "__main__":
    main()
