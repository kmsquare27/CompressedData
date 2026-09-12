"""Report observed training cost. No quality, test, inference, or FLOP claims."""
import argparse
from pathlib import Path
from sft_core import *

METRICS = """# Computational metrics in this report

- job_wall_seconds: per-arm script wall time after CLI parsing through final saving/telemetry shutdown; includes model loading, file verification, processing, optimization, logging and checkpoints. Excludes installation, model download and shared dataset preparation. This is training-job end-to-end latency, not inference latency.
- training_loop_wall_seconds: loop including checkpoint writes; training_loop_excluding_checkpoint_seconds subtracts separately timed checkpoint IO.
- update_wall_seconds: synchronized optimizer-update wall time, including image/HTML preparation, host-to-device transfers, forward/backward, gradient clipping, optimizer/scheduler and microbatch logging. Boundary synchronization is part of this instrumented recipe. This is not isolated GPU kernel time.
- sequence_tokens: actual processor-expanded input positions processed in completed optimizer windows. Includes visual placeholders, prompt, assistant HTML and assistant terminators, repeated for every epoch. It does not count gradient-checkpoint recomputation as additional data exposure.
- target_tokens: positions actually supervised after causal label shifting; prompt and image positions are masked. If BPE merges the template newline with leading HTML whitespace, that entire whitespace token is masked and the target whitespace character count is recorded in preparation. prompt_nonimage_tokens includes that masked boundary token. Non-whitespace boundary merges fail. HTML-only counts in preparation exclude assistant terminators, so they are a distinct unit.
- image_tokens: merged image-token positions in the language decoder. vision_patch_tokens counts pre-merge t*h*w patches fed to the vision encoder; it is a separate stream and is NOT added to sequence_tokens.
- padding_tokens: zero in this microbatch=1, no-packing recipe. Do not infer quadratic attention FLOPs or end-to-end cost from HTML-token saving alone.
- steady_* throughput/percentiles exclude the first configured warmup updates of THIS session. Full totals include them. Null means insufficient measurements; it never means zero cost.
- PyTorch peak allocated/reserved bytes: exact allocator peaks for the observed model-load/training interval, different from physical device occupancy. GiB = bytes / 2^30.
- NVML peak memory, utilization and energy: device-wide one-second samples, including other processes. Sampled peak can miss spikes. Energy is trapezoidal integration of available readings, without idle subtraction; report energy_covered_seconds and do not extrapolate missing intervals.
- RSS is sampled process host memory, not all node memory.
- failed/interrupted runs and resume sessions remain visible, but are excluded from canonical paired savings. Completed-backward events are separately retained in microbatches.jsonl; completed-update tokens in steps.jsonl omit unfinished updates.
- Resuming restores optimizer/scheduler/RNG/progress at an optimizer boundary. Replayed work since the last checkpoint has real operational cost; never sum logical cumulative counters across resume sessions. Use every session's logs for operational cost, and fresh uninterrupted reruns for the primary latency comparison.

Two epochs at 100 pages means 200 example presentations and 26 optimizer updates with accumulation=8 (12 windows of eight and one window of four per epoch). Example-normalized loss gives each example equal weight WITHIN its update; token-normalized loss weights by supervised token count within that update. No checkpoint is chosen from incomparable arm-specific loss values.
"""

def comparison_key(s):
    env = s["environment"]
    return digest({"config": s["config"], "data": s["data_fingerprint"], "revision": s["model_revision"],
                   "code": s["code_hashes"], "parent": s.get("parent_comparison_key"), "environment": {k:v for k,v in env.items() if k not in {"gpu_uuid", "platform"}}})

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", nargs="+", required=True, help="One or more run roots or arm session directories")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    files = set()
    for item in a.runs:
        root = Path(item)
        if root.is_file():
            files.add(root.resolve())
        else:
            files.update(p.resolve() for p in root.rglob("summary.json"))
    if not files:
        raise ValueError("No summary.json files found")
    table, groups, diagnostics = [], {}, []
    for path in sorted(files):
        s = read_json(path)
        steps = read_jsonl(path.with_name("steps.jsonl")) if path.with_name("steps.jsonl").exists() else []
        steady = [r for r in steps if not r["warmup_for_timing"]]
        total_wall = sum(r["update_wall_seconds"] for r in steps)
        steady_wall = sum(r["update_wall_seconds"] for r in steady)
        counts = {k:sum(r[k] for r in steps) for k in ["sequence_tokens", "target_tokens", "image_tokens", "prompt_nonimage_tokens", "vision_patch_tokens", "padding_tokens", "examples"]}
        issues = []
        if s["status"] == "completed":
            if len(steps) != s["completed_updates_this_session"]:
                issues.append("summary/step count mismatch")
            for k in s.get("session_tokens", {}):
                if counts[k] != s["session_tokens"][k]:
                    issues.append(f"summary/step {k} mismatch")
            if not s.get("smoke") and not s.get("resumed"):
                if counts["examples"] != s["pages"]*s["config"]["epochs"]:
                    issues.append("wrong example exposure count")
                expected_updates = math.ceil(s["pages"]/s["config"]["gradient_accumulation_steps"])*s["config"]["epochs"]
                if len(steps) != expected_updates:
                    issues.append("wrong optimizer-update count")
            if any(r["sequence_tokens"] != r["image_tokens"]+r["prompt_nonimage_tokens"]+r["target_tokens"] or r["padding_tokens"] != 0 for r in steps):
                issues.append("token partition/padding mismatch")
            if s.get("final_adapter_digest") == s.get("initial_adapter_digest"):
                issues.append("adapter unchanged")
        eligible = s["status"] == "completed" and not s.get("smoke") and not s.get("resumed") and not issues
        row = {"session": str(path.parent), "arm": s["arm"], "status": s["status"],
               "eligible_for_paired_comparison": eligible, "smoke": s.get("smoke"), "resumed": s.get("resumed"),
               "pages": s.get("pages"), "seed": s.get("config", {}).get("seed"),
               "optimizer_updates": len(steps), **counts,
               **{k:s.get(k) for k in ["job_wall_seconds", "model_load_seconds", "training_loop_wall_seconds", "training_loop_excluding_checkpoint_seconds", "checkpoint_seconds", "shared_preparation_seconds", "training_peak_allocated_bytes", "training_peak_reserved_bytes"]},
               "sum_update_wall_seconds": total_wall,
               "gpu_hours_training_job": s.get("job_wall_seconds",0)/3600,
               "job_seconds_per_training_example": s.get("job_wall_seconds",0)/counts["examples"] if counts["examples"] else None,
               "all_sequence_tokens_per_second": counts["sequence_tokens"]/total_wall if total_wall else None,
               "steady_updates": len(steady),
               "steady_sequence_tokens_per_second": sum(r["sequence_tokens"] for r in steady)/steady_wall if steady_wall else None,
               "steady_target_tokens_per_second": sum(r["target_tokens"] for r in steady)/steady_wall if steady_wall else None,
               "steady_examples_per_second": sum(r["examples"] for r in steady)/steady_wall if steady_wall else None,
               "steady_update_p50_seconds": float(np.percentile([r["update_wall_seconds"] for r in steady], 50)) if steady else None,
               "steady_update_p95_seconds": float(np.percentile([r["update_wall_seconds"] for r in steady], 95)) if steady else None,
               "issues": "; ".join(issues)}
        telemetry = s.get("telemetry", {})
        for phase in ["all", "train"]:
            for k,v in telemetry.get(phase, {}).items():
                row[f"nvml_{phase}_{k}"] = v
        table.append(row)
        if eligible:
            groups.setdefault(comparison_key(s), []).append(row)
        if issues:
            diagnostics.append({"session": str(path.parent), "issues": issues})
    write_csv(out / "compute_summary.csv", table)
    savings = []
    for group, rows in groups.items():
        by_arm = {arm:[r for r in rows if r["arm"] == arm] for arm in ARMS}
        if any(len(v) != 1 for v in by_arm.values()):
            diagnostics.append({"group": group, "issues": ["Paired saving requires exactly one original, naive and verified run per matched group; report repetitions separately"]})
            continue
        orig = by_arm["original"][0]
        for arm in ["naive", "verified"]:
            r = by_arm[arm][0]
            for metric in ["sequence_tokens", "target_tokens", "job_wall_seconds", "training_loop_excluding_checkpoint_seconds", "sum_update_wall_seconds", "training_peak_allocated_bytes", "training_peak_reserved_bytes"]:
                baseline, value = orig[metric], r[metric]
                savings.append({"group": group, "arm": arm, "metric": metric, "original": baseline,
                                "observed": value, "saving_pct": 100*(baseline-value)/baseline if baseline else None,
                                "speedup_ratio": baseline/value if value and "seconds" in metric else None})
        fig, axes = plt.subplots(1, 3, figsize=(13,4), constrained_layout=True)
        for ax, key, title, scale in zip(axes,
                ["sequence_tokens", "job_wall_seconds", "training_peak_allocated_bytes"],
                ["Processed sequence tokens", "Training job wall time (min)", "Training peak allocated (GiB)"],
                [1, 60, 2**30]):
            ax.bar(ARMS, [by_arm[arm][0][key]/scale for arm in ARMS], color=["#64748b", "#d97706", "#087f8c"])
            ax.set_title(title)
            ax.tick_params(axis="x", rotation=15)
            ax.spines[["top", "right"]].set_visible(False)
        fig.suptitle(f"Observed paired training cost — seed {orig['seed']}, {orig['pages']} pages")
        for ext in ["png", "pdf"]:
            fig.savefig(out / f"training_cost_{group[:10]}.{ext}", dpi=200)
        plt.close(fig)
    write_csv(out / "paired_savings.csv", savings)
    write_json(out / "report_issues.json", diagnostics)
    (out / "METRIC_DEFINITIONS.md").write_text(METRICS, encoding="utf-8")
    print(f"Reported {len(table)} sessions, {len(savings)} paired metric comparisons: {out}")

if __name__ == "__main__":
    main()
