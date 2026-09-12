"""CPU tests for data integrity and experimental accounting, not substitutes for GPU smoke."""
import argparse
import importlib.util
import json
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sft_core import *
spec = importlib.util.spec_from_file_location("exporter", ROOT / "12_export_sft_bundle.py")
exporter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exporter)

class TestProtocol(unittest.TestCase):
    def test_mask_only_assistant_and_preserve_eos(self):
        full = [101, 999, 102, 10, 11, 2]
        labels = mask_suffix(full, [101,999,102], 999)
        self.assertEqual(labels, [-100,-100,-100,10,11,2])
        self.assertEqual(sum(x != -100 for x in labels[1:]), 3)

    def test_bad_prefix_and_target_image_rejected(self):
        for full,prefix in [([1,2,3],[1,9]), ([1,2,999],[1,2]), ([1,2],[1,2])]:
            with self.assertRaises(ValueError):
                mask_suffix(full,prefix,999)

    def test_whitespace_boundary_masking_has_no_prompt_leak(self):
        labels, chars = mask_from_offsets([10,11,12], [(0,1),(1,3),(3,4)], 2, "A\n\nB",999)
        self.assertEqual(labels, [-100,-100,12])
        self.assertEqual(chars,1)
        with self.assertRaisesRegex(ValueError,"non-whitespace"):
            mask_from_offsets([10,11],[(0,1),(1,4)],2,"A\n<B",999)

    def test_loss_normalization_matches_definitions(self):
        counts, mean_losses = [2,6], [3.0,1.0]
        self.assertAlmostEqual(sum(w*l for w,l in zip(loss_weights(counts,"example"),mean_losses)), 2.0)
        self.assertAlmostEqual(sum(w*l for w,l in zip(loss_weights(counts,"token"),mean_losses)), 1.5)
        # Final partial window must normalize by four, not the configured accumulation of eight.
        self.assertEqual(loss_weights([1,2,3,4],"example"), [.25]*4)

    def test_deterministic_paired_order_and_complete_exposure(self):
        self.assertEqual(epoch_order(100,42,0), epoch_order(100,42,0))
        self.assertNotEqual(epoch_order(100,42,0), epoch_order(100,42,1))
        windows = [epoch_order(100,42,e)[s:s+8] for e in range(3) for s in range(0,100,8)]
        self.assertEqual(len(windows),39)
        self.assertEqual(sum(map(len,windows)),300)
        self.assertEqual([len(w) for w in windows].count(4),3)

    def test_lora_excludes_vision_and_embedding(self):
        names = ["model.language_model.layers.0.self_attn.q_proj", "model.language_model.layers.0.mlp.down_proj",
                 "model.visual.blocks.0.attn.qkv", "model.visual.layers.0.self_attn.q_proj", "lm_head"]
        self.assertEqual(allowed_lora_targets(names), names[:2])

    def test_portable_paths_and_nonfinite_json(self):
        with tempfile.TemporaryDirectory() as d:
            base = Path(d)
            with self.assertRaises(ValueError):
                portable_path(base,"../escape.html")
            write_json(base/"numbers.json", {"x":float("nan"),"y":float("inf")})
            self.assertEqual(read_json(base/"numbers.json"), {"x":None,"y":None})

class TestExport(unittest.TestCase):
    def fixture(self, root):
        rows, manifest, selection = [],[],[]
        for i in range(3):
            pid = str(i)
            d = root/f"inputs/{pid}"
            d.mkdir(parents=True)
            original = d/"original.html"
            final = d/"final.html"
            png = d/"image.png"
            original.write_bytes(b"<!doctype html>\r\n<p> hello </p>\r\n")
            final.write_bytes(b"<!doctype html><p>hello</p>")
            png.write_bytes(b"fixture-image-placeholder; exporter validates bytes, processor validates format")
            r = {"page_id":pid,"status":"ok","final_pixel_identical":True,
                 "final_level":"level1", "html_path":str(final),"png_path":str(png),
                 "original_sha256":file_sha(original),"final_sha256":file_sha(final),
                 "tokens_orig":20,"tokens_final":15}
            rows.append(r)
            manifest.append({k:v for k,v in r.items() if k not in {"status","final_pixel_identical","final_level"}} | {"level":"level1"})
            selection.append({"page_id":pid,"ladder":json.dumps([{"level":"original","html_path":str(original)}])})
        (root/"reports/csv").mkdir(parents=True)
        (root/"data/splits").mkdir(parents=True)
        write_csv(root/"reports/csv/final_validation_webcode2m.csv",rows)
        write_csv(root/"reports/csv/level_selection_webcode2m.csv",selection)
        write_csv(root/"data/splits/compressed_webcode2m_manifest.csv",manifest)
        write_json(root/"reports/csv/final_validation_webcode2m_environment.json",{"tolerance":False})
        return argparse.Namespace(root=str(root),source="webcode2m",n=3,seed=42,ids=None,out=str(root/"bundle"),zip=True)

    def test_export_is_portable_paired_and_byte_preserving(self):
        with tempfile.TemporaryDirectory() as d:
            args = self.fixture(Path(d))
            exporter.export(args)
            bundle = Path(args.out)
            rows = read_jsonl(bundle/"pages.jsonl")
            self.assertEqual(len(rows),3)
            self.assertEqual(validate_bundle_rows(bundle,rows),[])
            self.assertIn(b"\r\n",portable_path(bundle,rows[0]["original"]).read_bytes())
            with zipfile.ZipFile(str(bundle)+".zip") as z:
                self.assertIn("bundle/pages.jsonl",z.namelist())
            with self.assertRaises(ValueError):
                exporter.export(args)

    def test_mutations_and_csv_mismatches_collected_before_writes(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            args = self.fixture(root)
            (root/"inputs/0/original.html").write_text("changed")
            manifest = root/"data/splits/compressed_webcode2m_manifest.csv"
            rows = read_csv(manifest)
            rows[1]["tokens_final"] = 999
            write_csv(manifest,rows)
            with self.assertRaises(ValueError) as ctx:
                exporter.export(args)
            self.assertIn("original bytes changed",str(ctx.exception))
            self.assertIn("tokens_final mismatch",str(ctx.exception))
            self.assertFalse(Path(args.out).exists())

    def test_short_population_is_explicit_error(self):
        with tempfile.TemporaryDirectory() as d:
            args = self.fixture(Path(d))
            args.n = 100
            with self.assertRaisesRegex(ValueError,"Only 3 finalized pages"):
                exporter.export(args)

if __name__ == "__main__":
    unittest.main()
