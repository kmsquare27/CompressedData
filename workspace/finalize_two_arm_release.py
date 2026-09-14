"""Finalize the supplied repaired SFT project for original/verified comparisons.

Run from Windows or Linux with Python 3.11/3.12.
Tests require numpy and matplotlib in the executing Python environment.
Does not train, download models, or modify datasets/checkpoints.
"""
from __future__ import annotations

import ast
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import textwrap
import zipfile

ROOT = (
    Path(sys.argv[1])
    if len(sys.argv) > 1
    else Path(__file__).resolve().parent / "qwen_sft_patch"
).resolve()

if not (ROOT / "sft_core.py").is_file():
    raise SystemExit(f"Project not found: {ROOT}")

changes = {}


def source(name):
    if name in changes:
        return changes[name]
    return (ROOT / name).read_text(encoding="utf-8")


def replace(name, old, new, expected=1):
    text = source(name)
    count = text.count(old)
    if count != expected:
        raise RuntimeError(
            f"{name}: expected {expected} matching block(s), found {count}. "
            "Project files have not been changed."
        )
    changes[name] = text.replace(old, new)


def regex_replace(name, pattern, replacement, expected=1):
    text, count = re.subn(pattern, replacement, source(name))
    if count != expected:
        raise RuntimeError(
            f"{name}: expected {expected} matching pattern(s), found {count}. "
            "Project files have not been changed."
        )
    changes[name] = text


# Verify that this is the repaired project, not the original pod version.
core_tree = ast.parse(source("sft_core.py"))
functions = {
    node.name for node in core_tree.body
    if isinstance(node, ast.FunctionDef)
}
required = {
    "checkpoint_dirs",
    "prune_checkpoints",
    "checkpoint_schedule",
    "checkpoint_disk_plan",
    "verify_training_checkpoint",
}
if not required <= functions:
    raise SystemExit(
        "This is not the repaired project. Missing helpers: "
        + ", ".join(sorted(required - functions))
    )

stage1 = json.loads(source("configs/stage1_2250_12k.json"))
stage2 = json.loads(source("configs/stage2_7500_12k.json"))
continuation_keys = (
    "model_id", "seed", "lora_r", "lora_alpha", "lora_dropout",
    "max_seq_length", "min_image_tokens", "max_image_tokens",
    "gradient_accumulation_steps", "learning_rate", "weight_decay",
    "warmup_ratio", "loss_normalization", "max_grad_norm", "epochs",
)
for key in continuation_keys:
    if key not in stage1 or key not in stage2 or stage1[key] != stage2[key]:
        raise SystemExit(f"Stage configuration mismatch: {key}")


# ---------------------------------------------------------------------------
# Step 15: retain existing accounting; select comparison arms explicitly.
# ---------------------------------------------------------------------------

REPORT = "15_report_sft_compute.py"

replace(
    REPORT,
    '    a = p.parse_args()\n',
    '''    p.add_argument(
        "--arms", nargs="+", choices=ARMS,
        default=["original", "verified"],
        help="Comparison arms; defaults to original verified."
    )
    a = p.parse_args()
    if (
        len(a.arms) < 2
        or len(set(a.arms)) != len(a.arms)
        or "original" not in a.arms
    ):
        p.error("Choose original and at least one comparison arm, without duplicates")
    arms = [arm for arm in ARMS if arm in a.arms]
    comparison_arms = [arm for arm in arms if arm != "original"]
''',
)

replace(
    REPORT,
    'by_arm = {arm:[r for r in rows if r["arm"] == arm] for arm in ARMS}',
    'by_arm = {arm:[r for r in rows if r["arm"] == arm] for arm in arms}',
)

replace(
    REPORT,
    '["Paired saving requires exactly one original, naive and verified run per matched group; report repetitions separately"]',
    '[f"Paired saving requires exactly one eligible run for each selected arm '
    '({\', \'.join(arms)}) per matched group; report repetitions separately"]',
)

regex_replace(
    REPORT,
    r'for arm in \[\s*"naive"\s*,\s*"verified"\s*\]:',
    "for arm in comparison_arms:",
)

replace(
    REPORT,
    '            ax.bar(ARMS, [by_arm[arm][0][key]/scale for arm in ARMS], color=["#64748b", "#d97706", "#087f8c"])\n',
    '''            colors = {
                "original": "#64748b",
                "naive": "#d97706",
                "verified": "#087f8c",
            }
            ax.bar(
                arms,
                [by_arm[arm][0][key] / scale for arm in arms],
                color=[colors[arm] for arm in arms],
            )
''',
)

replace(
    REPORT,
    '(out / "METRIC_DEFINITIONS.md").write_text(METRICS, encoding="utf-8")',
    '(out / "METRIC_DEFINITIONS.md").write_text(\n'
    '        METRICS + "\\nSelected comparison arms: " + ", ".join(arms)\n'
    '        + ". Raw session rows may include unselected arms.\\n",\n'
    '        encoding="utf-8",\n'
    '    )',
)


# ---------------------------------------------------------------------------
# Step 18: keep the actual root-level layout and support two or three profiles.
# ---------------------------------------------------------------------------

INFERENCE = "18_inference_profile.py"

replace(
    INFERENCE,
    "Install in qwen_sft_patch/compute_tools; imports the existing parent sft_core.",
    "Keep beside sft_core.py in the qwen_sft_patch project root.",
)

replace(
    INFERENCE,
    "sys.path.insert(0, str(Path(__file__).resolve().parent.parent))",
    "PROJECT_ROOT = Path(__file__).resolve().parent\n"
    "sys.path.insert(0, str(PROJECT_ROOT))",
)

regex_replace(
    INFERENCE,
    r'Path\(__file__\)\.resolve\(\)\.parent\.parent\s*/\s*"sft_core\.py"',
    'PROJECT_ROOT / "sft_core.py"',
)

replace(
    INFERENCE,
    '''    profiles = [read_json(Path(x)/"profile.json") for x in paths]
    if len(profiles)!=3 or {s["arm"] for s in profiles}!=set(ARMS):
        raise ValueError("Supply exactly one profile directory for each arm")
''',
    '''    if len(paths) not in {2, 3}:
        raise ValueError("Supply original plus verified and/or naive profiles")
    profiles = [read_json(Path(x)/"profile.json") for x in paths]
    supplied_arms = [profile.get("arm") for profile in profiles]
    if (
        len(set(supplied_arms)) != len(supplied_arms)
        or "original" not in supplied_arms
        or not set(supplied_arms) <= set(ARMS)
    ):
        raise ValueError(
            "Supply one original profile and one per selected comparison arm; "
            "unknown or duplicate arms are not allowed"
        )
    comparison_arms = [
        arm for arm in ARMS if arm != "original" and arm in supplied_arms
    ]
''',
)

regex_replace(
    INFERENCE,
    r'for arm in \[\s*"naive"\s*,\s*"verified"\s*\]:',
    "for arm in comparison_arms:",
    expected=2,
)

replace(
    INFERENCE,
    '    p.add_argument("--compare",nargs=3,metavar="PROFILE_DIR")',
    '    p.add_argument("--compare", nargs="+", metavar="PROFILE_DIR",\n'
    '                   help="Two or three profiles: original plus verified and/or naive")',
)

replace(
    INFERENCE,
    '        quality_evaluated=False,note="Truncated and low-quality generations remain included;',
    '        arms=["original"] + comparison_arms,\n'
    '        quality_evaluated=False,note="Truncated and low-quality generations remain included;',
)


# ---------------------------------------------------------------------------
# Tests: real reporting subprocesses with synthetic fixtures; no GPU work.
# ---------------------------------------------------------------------------

changes["tests/test_final_pair_release.py"] = textwrap.dedent(r'''
import importlib.util
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import sft_core as core


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


old_report_tests = load("report_fixtures", ROOT / "tests/test_report.py")
inference = load("pair_inference", ROOT / "18_inference_profile.py")


class PairReportTests(unittest.TestCase):
    def report(self, runs, output, *extra):
        subprocess.run(
            [
                sys.executable, str(ROOT / "15_report_sft_compute.py"),
                "--runs", str(runs), "--out", str(output), *extra,
            ],
            check=True, capture_output=True, text=True,
        )

    def test_pair_succeeds_without_naive(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            old_report_tests.TestReport().fixture(root / "runs")
            shutil.rmtree(root / "runs/naive")
            self.report(root / "runs", root / "report")
            savings = core.read_csv(root / "report/paired_savings.csv")
            self.assertEqual(len(savings), 7)
            self.assertEqual({row["arm"] for row in savings}, {"verified"})
            wall = next(row for row in savings if row["metric"] == "job_wall_seconds")
            self.assertEqual(float(wall["saving_pct"]), 40.0)
            self.assertEqual(len(list((root / "report").glob("*.png"))), 1)

    def test_explicit_three_arm_support_is_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            old_report_tests.TestReport().fixture(root / "runs")
            self.report(
                root / "runs", root / "report",
                "--arms", "original", "naive", "verified",
            )
            self.assertEqual(
                len(core.read_csv(root / "report/paired_savings.csv")), 14
            )

    def test_missing_selected_arm_prevents_pairing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            old_report_tests.TestReport().fixture(root / "runs")
            shutil.rmtree(root / "runs/verified")
            self.report(root / "runs", root / "report")
            self.assertEqual(
                core.read_csv(root / "report/paired_savings.csv"), []
            )


class PairInferenceTests(unittest.TestCase):
    def fixture(self, root):
        paths = []
        for arm, tokens in (("original", 10), ("verified", 5)):
            folder = root / arm
            folder.mkdir()
            row = {
                "repeat": 0, "page_id": "p", "status": "completed",
                "image_sha256": "same", "prompt_tokens": 10, "image_tokens": 4,
                "request_seconds": 2, "generation_seconds": 1,
                "ttft_generate_seconds": 0.2, "generated_tokens": tokens,
                "peak_allocated_bytes": 100,
                "truncated_by_limit": arm == "verified",
            }
            core.append_jsonl(folder / "requests.jsonl", row)
            core.write_json(folder / "profile.json", {
                "arm": arm, "eligible_for_complete_comparison": True,
                "comparison_key": "same", "page_ids": ["p"],
                "arguments": {"repeats": 1},
                "requests_sha256": core.file_sha(folder / "requests.jsonl"),
            })
            paths.append(str(folder))
        return paths

    def test_pair_preserves_truncated_requests(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = self.fixture(root)
            inference.compare_profiles(paths, root / "report")
            rows = core.read_csv(root / "report/paired_inference.csv")
            self.assertEqual(len(rows), 5)
            self.assertTrue(all(row["either_truncated"] == "True" for row in rows))
            totals = core.read_csv(root / "report/inference_savings.csv")
            tokens = next(row for row in totals if row["metric"] == "generated_tokens")
            self.assertEqual(float(tokens["saving_pct"]), 50.0)

    def test_duplicate_and_changed_ledger_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = self.fixture(root)
            with self.assertRaises(ValueError):
                inference.compare_profiles([paths[0], paths[0]], root / "bad")
            with (Path(paths[0]) / "requests.jsonl").open("a") as stream:
                stream.write("\n")
            with self.assertRaises(ValueError):
                inference.compare_profiles(paths, root / "changed")

    def test_root_location_and_cli_outside_project(self):
        self.assertEqual(inference.PROJECT_ROOT, ROOT)
        self.assertTrue((inference.PROJECT_ROOT / "sft_core.py").is_file())
        with tempfile.TemporaryDirectory() as directory:
            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "18_inference_profile.py"),
                    "--help",
                ],
                cwd=directory, check=True, capture_output=True, text=True,
            )


if __name__ == "__main__":
    unittest.main()
''').lstrip()


# Historical guides remain available, with an explicit notice.
historical_notice = (
    "> HISTORICAL GUIDE: its three-arm, pilot, path and sequence-limit commands "
    "are superseded for the current experiment. "
    "Read CURRENT_RUN.md before executing commands.\n\n"
)
for name in (
    "README.md", "PATCH_NOTES.md", "README_STEPS_17_18.md", "TESTING.md"
):
    changes[name] = historical_notice + source(name)

changes["CURRENT_RUN.md"] = """# Current two-arm SFT release

## Active decisions

- Independently initialize original and verified from the locked
  Qwen/Qwen2.5-VL-7B-Instruct base.
- Do not initialize from pilot adapters or train a naive arm.
- Source dataset: /workspace/html_compression_linux_v2.
- Stage-one page count comes from the audited export and preparation.
  The number 2250 in a configuration filename does not set that count.
- Later continue each corresponding stage-one adapter on disjoint new pages.
- Keep existing code, datasets, model files and pilot results preserved.

## Active layout

Scripts 00 through 18, sft_core.py and telemetry.py are at the project root.
There is no required compute_tools subdirectory in this release.

Deploy the complete finalized ZIP to a new directory, for example:
  /workspace/sft_ready_release/qwen_sft_patch

Reuse:
  /workspace/sft-venv
  /workspace/models/qwen7b
  /workspace/sft_artifacts/qwen7b_uploaded_model_lock.json

Do not rerun old overlay/repair installers on this finalized tree.

## Configurations

Stage one: configs/stage1_2250_12k.json
Stage two: configs/stage2_7500_12k.json

Both use a 12288 expanded sequence cap, subject to full-dataset preflight.
They retain three checkpoints per session and save at epoch ends as well as
every 50/100 updates respectively. Final adapters are saved separately.

## Before training

1. Verify the shared-volume mount, available storage and locked model.
2. Audit Step 08 records, referenced files, acceptance policy and excluded IDs.
3. Export exact audited IDs and preserve valid original fallbacks.
4. Prepare original/verified together with no silent truncation.
5. Inspect exclusions, actual processed lengths and image legibility.
6. Run the matched GPU smoke tests.
7. Check actual interruption recovery and parent-adapter reload on the GPU.

Strict pixel identity is the export default. The validated policy accepts
Step 08 status=ok without adding that stricter pixel rule. Choose based on
the saved Linux experiment, not a desired page count.

## Initialization

Fresh stage one:
  run_all_arms.py ... --arms original verified

No --resume or --init-from-runs argument belongs on the fresh stage-one command.

New-data continuation:
  run_all_arms.py ... --arms original verified --init-from-runs STAGE_ONE_RUN_ROOT

Parent smoke:
  use the same parent argument with --smoke and a separate diagnostic output.

Actual continuation uses the stage-one parent again, never the smoke output.
It starts a fresh optimizer and schedule.

Interruption recovery:
  14_train_sft.py ... --arm ARM --resume ACTUAL_CHECKPOINT --out NEW_SESSION

Use the same stage data/config/code for resume. A final-update checkpoint can
recover final saving without repeating training updates.

## Verification and reporting

Final adapter:
  python 16_verify_sft_checkpoint.py --run ARM_SESSION

Intermediate checkpoint files:
  python 16_verify_sft_checkpoint.py --checkpoint CHECKPOINT_DIRECTORY

Training comparison:
  python 15_report_sft_compute.py --runs RUN_ROOT --out NEW_REPORT --arms original verified

Inference comparison, when evaluation is ready:
  python 18_inference_profile.py --compare ORIGINAL_PROFILE VERIFIED_PROFILE --out NEW_REPORT

Inference generation still requires a completed adapter run; this patch does
not add base-model evaluation or reconstruction-quality scoring.

## Validation boundary

The finalizer executes focused CPU reporting and inference-contract tests in
a temporary copy before applying its changes. Synthetic fixture values are
not experimental results. Real dataset compatibility, CUDA execution,
processor behavior and checkpoint restoration remain pod checks.

Historical test records in other documents do not certify this release.
"""


# Compile changes before creating any project files.
for name, text in changes.items():
    if name.endswith(".py"):
        compile(text, str(ROOT / name), "exec")

stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")

# Tests execute on a copy; any failure leaves the real project unchanged.
with tempfile.TemporaryDirectory(prefix="sft_pair_check_") as temporary:
    staged = Path(temporary) / "qwen_sft_patch"
    shutil.copytree(
        ROOT, staged,
        ignore=shutil.ignore_patterns(
            "__pycache__", "*.pyc", ".git", ".venv", "venv",
            "*.zip", "*.tar.gz",
        ),
    )
    for name, text in changes.items():
        path = staged / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8", newline="\n")

    for pattern in (
        "test_final_pair_release.py",
        "test_report.py",
        "test_compute_extensions.py",
    ):
        subprocess.run(
            [
                sys.executable, "-m", "unittest", "discover",
                "-s", "tests", "-p", pattern, "-v",
            ],
            cwd=staged,
            check=True,
        )

backup = ROOT.parent / f"sft_before_pair_finalization_{stamp}.tar.gz"
names_to_preserve = set(changes) | {"SHA256SUMS.txt"}
originals = {
    name: (ROOT / name).read_bytes() if (ROOT / name).is_file() else None
    for name in names_to_preserve
}
with tarfile.open(backup, "x:gz") as archive:
    for name, payload in originals.items():
        if payload is not None:
            archive.add(ROOT / name, arcname=f"qwen_sft_patch/{name}")

suffixes = {".py", ".sh", ".json", ".md", ".txt", ".toml", ".yaml", ".yml"}


def source_files():
    return sorted(
        path for path in ROOT.rglob("*")
        if path.is_file()
        and not path.is_symlink()
        and path.suffix in suffixes
        and not any(
            part.startswith(".") or part in {"__pycache__", "venv"}
            for part in path.relative_to(ROOT).parts
        )
    )


try:
    for name, text in changes.items():
        path = ROOT / name
        path.parent.mkdir(parents=True, exist_ok=True)
        pending = path.with_name(path.name + ".pair-tmp")
        pending.write_text(text, encoding="utf-8", newline="\n")
        os.replace(pending, path)

    checksums = []
    for path in source_files():
        if path.name == "SHA256SUMS.txt":
            continue
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        checksums.append(f"{sha}  {path.relative_to(ROOT).as_posix()}")
    (ROOT / "SHA256SUMS.txt").write_text(
        "\n".join(checksums) + "\n", encoding="utf-8", newline="\n"
    )
except BaseException:
    for name, payload in originals.items():
        path = ROOT / name
        if payload is None:
            path.unlink(missing_ok=True)
        else:
            path.write_bytes(payload)
    raise

zip_path = ROOT.parent / f"qwen_sft_final_two_arms_{stamp}.zip"
with zipfile.ZipFile(
    zip_path, "x", compression=zipfile.ZIP_DEFLATED
) as archive:
    for path in source_files():
        archive.write(
            path, f"qwen_sft_patch/{path.relative_to(ROOT).as_posix()}"
        )

with zipfile.ZipFile(zip_path) as archive:
    bad = archive.testzip()
    if bad:
        raise RuntimeError(f"ZIP integrity check failed: {bad}")

print("\nFinalization completed after staged CPU tests passed.")
print("Backup:", backup)
print("UPLOAD THIS ZIP:", zip_path)
print("Trainer, exporter, preparation code, core helpers and configs were not changed.")