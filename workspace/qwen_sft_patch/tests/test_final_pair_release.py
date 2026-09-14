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
