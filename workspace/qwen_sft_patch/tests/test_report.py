"""Synthetic integration tests of the report; values are fixtures, never research results."""
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from sft_core import *

class TestReport(unittest.TestCase):
    def fixture(self, folder, corrupt=False):
        config = load_config(ROOT/"configs/pilot_100.json") | {"epochs":1}
        for arm, sequence, seconds in [("original",100,10),("naive",70,8),("verified",60,6)]:
            out = folder/arm
            out.mkdir(parents=True)
            counts = {"sequence_tokens":sequence,"target_tokens":sequence-20,"image_tokens":10,
                      "prompt_nonimage_tokens":10,"vision_patch_tokens":40,"padding_tokens":0}
            step = {"global_step":1,"epoch":1,"examples":2,**counts,"update_wall_seconds":seconds-2,
                    "warmup_for_timing":False}
            write_jsonl(out/"steps.jsonl",[step])
            summary = {"status":"completed","arm":arm,"smoke":False,"resumed":False,"pages":2,
                       "config":config,"completed_updates_this_session":1,"session_tokens":dict(counts),
                       "final_adapter_digest":"changed","initial_adapter_digest":"initial",
                       "job_wall_seconds":seconds,"model_load_seconds":1,"training_loop_wall_seconds":seconds-2,
                       "training_loop_excluding_checkpoint_seconds":seconds-2,"checkpoint_seconds":0,
                       "shared_preparation_seconds":3,"training_peak_allocated_bytes":2**30,
                       "training_peak_reserved_bytes":2**31,"model_revision":"a"*40,
                       "data_fingerprint":"fixture","code_hashes":{"trainer":"fixture"},
                       "environment":{"gpu":"SYNTHETIC FIXTURE","packages":{}}}
            if corrupt and arm=="verified":
                summary["session_tokens"]["sequence_tokens"] += 1
            write_json(out/"summary.json",summary)

    def run_report(self, folder, out):
        subprocess.run([sys.executable,str(ROOT/"15_report_sft_compute.py"),"--runs",str(folder),"--out",str(out)],
                       check=True,capture_output=True,text=True)

    def test_exact_paired_savings_and_figures(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            self.fixture(root/"runs")
            self.run_report(root/"runs",root/"report")
            rows = read_csv(root/"report/paired_savings.csv")
            value = next(r for r in rows if r["arm"]=="verified" and r["metric"]=="job_wall_seconds")
            self.assertEqual(float(value["saving_pct"]),40.0)
            self.assertAlmostEqual(float(value["speedup_ratio"]),10/6)
            self.assertEqual(len(list((root/"report").glob("*.png"))),1)
            self.assertEqual(len(list((root/"report").glob("*.pdf"))),1)

    def test_counter_mismatch_excludes_comparison(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            self.fixture(root/"runs",corrupt=True)
            self.run_report(root/"runs",root/"report")
            self.assertEqual(read_csv(root/"report/paired_savings.csv"),[])
            self.assertTrue(read_json(root/"report/report_issues.json"))

if __name__ == "__main__":
    unittest.main()
