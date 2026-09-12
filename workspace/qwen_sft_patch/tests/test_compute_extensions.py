"""CPU tests for measurement contracts; no claims about GPU execution."""
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]


def module(name):
    spec = importlib.util.spec_from_file_location(name, ROOT/(name+".py"))
    obj = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj)
    return obj


cost = module("17_compression_cost")
inf = module("18_inference_profile")


class Tensor:
    def __init__(self, values): self.values = values
    def detach(self): return self
    def cpu(self): return self
    def reshape(self, *args): return self
    def tolist(self): return self.values


class CostTests(unittest.TestCase):
    def test_break_even(self):
        self.assertEqual(cost.break_even(600,60)["epochs"],10)
        self.assertIsNone(cost.break_even(600,0)["epochs"])
        self.assertIsNone(cost.break_even(600,-2)["epochs"])
        self.assertEqual(cost.break_even(-1,10)["epochs"],0)

    def stage(self, **kw):
        return dict(kind="stage",run_id="r",variant="verified",stage="l1",status="completed",
                    page_ids_sha256="a",pages=2,elapsed_seconds=10,
                    started_utc="2026-01-01T00:00:00+00:00",hourly_rate=3.6,currency="USD") | kw

    def test_stage_scope_and_cost(self):
        r = cost.stage_totals([self.stage(),dict(kind="page",elapsed_seconds=999)],"r","verified",["l1"])
        self.assertEqual(r["seconds"],10)
        self.assertAlmostEqual(r["cost"],.01)

    def test_fail_duplicate_missing_overlap(self):
        for rows,stages in [([self.stage(status="failed")],["l1"]),
                            ([self.stage(),self.stage()],["l1"]),
                            ([self.stage()],["l1","l2"]),
                            ([self.stage(),self.stage(stage="l2")],["l1","l2"])]:
            with self.assertRaises(ValueError): cost.stage_totals(rows,"r","verified",stages)

    def test_actual_command_and_failure_preserved(self):
        with tempfile.TemporaryDirectory() as d:
            ids = Path(d)/"ids.txt"
            ids.write_text("one\ntwo\n")
            a = SimpleNamespace(command=["--",sys.executable,"-c","raise SystemExit(3)"],page_ids=ids,
                run_id="r",stage="test",variant="verified",hardware_label="test",cwd=d,
                hourly_rate=None,currency="USD",ledger=Path(d)/"ledger.jsonl")
            self.assertEqual(cost.record(a),3)
            row=json.loads(a.ledger.read_text())
            self.assertEqual(row["status"],"failed")
            self.assertEqual(row["pages"],2)
            self.assertGreater(row["elapsed_seconds"],0)

    def test_page_exception_not_suppressed(self):
        with tempfile.TemporaryDirectory() as d:
            log = Path(d)/"pages.jsonl"
            with self.assertRaises(RuntimeError):
                with cost.StageTimer(log,run_id="r",stage="l1",page_id="p"):
                    raise RuntimeError("bad page")
            self.assertEqual(json.loads(log.read_text())["status"],"failed")

    def test_complete_cost_report(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            ledger=root/"ledger.jsonl"
            cost.append(ledger,self.stage(variant="original",stage="baseline",elapsed_seconds=10))
            cost.append(ledger,self.stage(stage="l1",elapsed_seconds=110))
            paths=[]
            for arm,seconds in [("original",100),("verified",80)]:
                folder=root/arm
                folder.mkdir()
                s=dict(status="completed",smoke=False,resumed=False,arm=arm,config=dict(epochs=2),
                    data_fingerprint="d",model_revision="r",code_hashes={"x":"h"},
                    environment=dict(gpu="test"),pages=2,
                    training_loop_excluding_checkpoint_seconds=seconds,job_wall_seconds=seconds+5)
                (folder/"summary.json").write_text(json.dumps(s))
                paths.append(folder/"summary.json")
            inf.write_csv(root/"compute.csv",[dict(session=str(x.parent),eligible_for_paired_comparison=True) for x in paths])
            a=SimpleNamespace(ledger=ledger,run_id="r",verified_stages=["l1"],original_stages=["baseline"],
                original_summary=paths[0],verified_summary=paths[1],compute_summary=root/"compute.csv",
                scope_note="all two pages",gpu_hourly_rate=3.6,currency="USD",out=root/"out.json")
            cost.report(a)
            report=cost.read_json(a.out)
            self.assertAlmostEqual(report["elapsed_break_even"]["epochs"],10)
            self.assertAlmostEqual(report["money_break_even"]["epochs"],10)


class InferenceTests(unittest.TestCase):
    def test_clock_skips_prompt_and_records_tokens(self):
        times=iter([11,13])
        clock=inf.TokenClock([1,2],clock=lambda:next(times))
        clock.put(Tensor([1,2]))
        clock.put(Tensor([3]))
        clock.put(Tensor([9]))
        m=inf.token_metrics(clock.ids,clock.times,10,14,[9],10)
        self.assertEqual(m["ttft_generate_seconds"],1)
        self.assertEqual(m["decode_tokens_per_second"],.5)
        self.assertEqual(m["generated_tokens_excluding_terminal_eos"],1)
        self.assertFalse(m["truncated_by_limit"])

    def test_stream_contract_rejected(self):
        with self.assertRaises(ValueError): inf.TokenClock([1]).put(Tensor([2]))
        clock=inf.TokenClock([1])
        clock.put(Tensor([1]))
        with self.assertRaises(ValueError): clock.put(Tensor([2,3]))

    def test_eos_at_cap_and_one_token(self):
        m=inf.token_metrics([9],[11],10,12,[9],1)
        self.assertTrue(m["hit_output_limit"])
        self.assertFalse(m["truncated_by_limit"])
        self.assertIsNone(m["decode_tokens_per_second"])
        self.assertTrue(inf.token_metrics([2],[11],10,12,[9],1)["truncated_by_limit"])

    def test_bad_times(self):
        with self.assertRaises(ValueError): inf.token_metrics([1,2],[12,11],10,13,[9],4)

    def test_html_preservation(self):
        html="<html>\n<p>x</p>\n</html>"
        self.assertEqual(inf.html_text(html),(html,False))
        self.assertEqual(inf.html_text("```html\n"+html+"\n```"),(html,True))
        prose="Here it is:\n```html\n"+html+"\n```"
        self.assertEqual(inf.html_text(prose),(prose,False))

    def test_heldout_overlap_and_hashes(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            row=dict(page_id="p")
            for k in ["image","original","verified"]:
                (root/k).write_bytes(k.encode())
                row[k]=k
                row[k+"_sha256"]=inf.file_sha(root/k)
            (root/"pages.jsonl").write_text(json.dumps(row)+"\n")
            inf.write_json(root/"bundle.json",dict(ready=True,pages=1,pages_sha256=inf.file_sha(root/"pages.jsonl")))
            with self.assertRaises(ValueError): inf.held_out(root,dict(training_page_ids=[],ancestor_page_ids=["p"]),0,42)
            rows,_=inf.held_out(root,dict(training_page_ids=["other"]),0,42)
            self.assertEqual(rows[0]["page_id"],"p")
            (root/"image").write_bytes(b"changed")
            with self.assertRaises(ValueError): inf.held_out(root,dict(training_page_ids=[]),0,42)

    def test_compare_and_tampered_ledger(self):
        with tempfile.TemporaryDirectory() as d:
            paths=[]
            for arm in ["original","naive","verified"]:
                root=Path(d)/arm
                root.mkdir()
                row=dict(repeat=0,page_id="p",status="completed",image_sha256="a",prompt_tokens=10,
                    image_tokens=4,request_seconds=2,generation_seconds=1,ttft_generate_seconds=.2,
                    generated_tokens=10 if arm=="original" else 5,peak_allocated_bytes=100,truncated_by_limit=False)
                inf.append_jsonl(root/"requests.jsonl",row)
                inf.write_json(root/"profile.json",dict(arm=arm,eligible_for_complete_comparison=True,
                    comparison_key="same",page_ids=["p"],arguments=dict(repeats=1),
                    requests_sha256=inf.file_sha(root/"requests.jsonl")))
                paths.append(str(root))
            inf.compare_profiles(paths,Path(d)/"report")
            self.assertIn("50.0",(Path(d)/"report/inference_savings.csv").read_text())
            with (Path(paths[0])/"requests.jsonl").open("a") as f: f.write("\n")
            with self.assertRaises(ValueError): inf.compare_profiles(paths,Path(d)/"other")


if __name__ == "__main__":
    unittest.main()
