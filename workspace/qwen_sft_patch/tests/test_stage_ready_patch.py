import argparse
import contextlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import sft_core as core


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


exporter = module("patched_exporter", ROOT / "12_export_sft_bundle.py")
preparer = module("patched_preparer", ROOT / "13_prepare_sft.py")
historical = module("historical_export_tests", ROOT / "tests/test_core_and_export.py")


class Checkpoints(unittest.TestCase):
    def fixture(self, root, step):
        recipe = {"fixture": "same-stage"}
        recipe_hash = core.digest(recipe)
        core.write_json(root / "recipe.json", {"recipe_hash": recipe_hash})
        folder = root / "checkpoints" / f"step_{step:06d}"
        folder.mkdir(parents=True)
        core.write_json(folder / "stage_recipe.json", recipe)
        core.write_json(folder / "adapter_config.json", {"r": 64})
        (folder / "adapter_model.safetensors").write_bytes(b"fixture")
        (folder / "training_state.pt").write_bytes(b"fixture")
        hashes = {p.name: core.file_sha(p) for p in folder.iterdir()}
        core.write_json(folder / "checkpoint.json", {
            "schema": 2, "recipe_hash": recipe_hash,
            "files_sha256": hashes,
            "trainable_parameters": [],
            "progress": {"global_step": step, "next_window": step},
        })
        return folder, recipe_hash

    def test_retention_ignores_partial_foreign_and_symlink(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for step in (1, 2, 10):
                latest, recipe_hash = self.fixture(root, step)
            parent = root / "checkpoints"
            for name in ("step_999999.incomplete", "foreign"):
                folder = parent / name
                folder.mkdir()
                core.write_json(folder / "checkpoint.json", {})
            (parent / "step_000011").symlink_to(latest, target_is_directory=True)
            removed, retained = core.prune_checkpoints(root, 2, recipe_hash)
            self.assertEqual(len(removed), 1)
            self.assertEqual([Path(x).name for x in retained],
                             ["step_000002", "step_000010"])
            self.assertTrue((parent / "step_999999.incomplete").exists())
            self.assertTrue((parent / "foreign").exists())
            self.assertTrue((parent / "step_000011").is_symlink())

    def test_config_corruption_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            folder, recipe_hash = self.fixture(Path(directory), 3)
            core.verify_training_checkpoint(folder, recipe_hash)
            (folder / "adapter_config.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "adapter_config"):
                core.verify_training_checkpoint(folder, recipe_hash)

    def test_schedule_and_finalization_disk_budget(self):
        self.assertEqual(len(core.checkpoint_schedule(560, 280, 2, 50)), 13)
        plan = core.checkpoint_disk_plan(100, 560, 280, 2, 50, 3)
        self.assertEqual(plan["retained_checkpoints"], 3)
        self.assertGreaterEqual(plan["peak_run_bytes"], 4 * 1200 + 800)
        final = core.checkpoint_disk_plan(
            100, 560, 280, 2, 50, 3, start_step=560
        )
        self.assertEqual(final["planned_checkpoints"], 0)
        self.assertGreater(final["peak_run_bytes"], 0)


class ExportPolicies(unittest.TestCase):
    def fixture(self, root):
        args = historical.TestExport().fixture(root)
        args.n = None
        setattr(args, "all", True)
        args.zip = False
        return args

    def test_missing_validated_manifest_row_fails_with_audit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = self.fixture(root)
            manifest = root / "data/splits/compressed_webcode2m_manifest.csv"
            core.write_csv(manifest, core.read_csv(manifest)[1:])
            with self.assertRaisesRegex(ValueError, "no compressed-manifest"):
                exporter.export(args)
            self.assertTrue((Path(args.out) / "export_issues.json").is_file())
            self.assertFalse((Path(args.out) / "bundle.json").exists())

    def test_strict_and_validated_policies_are_explicit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = self.fixture(root)
            report = root / "reports/csv/final_validation_webcode2m.csv"
            rows = core.read_csv(report)
            rows[0]["final_pixel_identical"] = False
            core.write_csv(report, rows)
            args.plan_only = True
            exporter.export(args)
            audit = core.read_json(Path(args.out) / "reconciliation_summary.json")
            self.assertEqual(audit["selected_pages"], 2)
            self.assertTrue((Path(args.out) / "page_ids.txt").is_file())
            args.out = str(root / "validated_plan")
            args.acceptance_policy = "validated"
            exporter.export(args)
            audit = core.read_json(Path(args.out) / "reconciliation_summary.json")
            self.assertEqual(audit["selected_pages"], 3)

    def test_empty_id_selection_never_becomes_ready(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = self.fixture(root)
            setattr(args, "all", False)
            path = root / "empty.txt"
            path.write_text("")
            args.ids = str(path)
            with self.assertRaisesRegex(ValueError, "No pages selected"):
                exporter.export(args)
            self.assertFalse((Path(args.out) / "bundle.json").exists())

    def test_equivalent_original_paths_are_deduplicated(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            row = {"ladder": json.dumps([
                {"level": "original", "html_path": "inputs/a.html"},
                {"level": "original", "html_path": "inputs/../inputs/a.html"},
            ])}
            self.assertEqual(
                exporter.original_from_ladder(root, row),
                (root / "inputs/a.html").resolve(),
            )


class Vector(list):
    def tolist(self):
        return list(self)


class Preparation(unittest.TestCase):
    def run_fixture(self, root, expect, long_page=False, budget=None, corrupt=False):
        bundle = root / "bundle"
        bundle.mkdir()
        rows = []
        for index in range(2):
            folder = bundle / str(index)
            folder.mkdir()
            row = {"page_id": str(index), "final_level": "l1"}
            for key in ("original", "verified", "image"):
                path = folder / key
                payload = (
                    b"LONG" if long_page and index == 0 and key == "original"
                    else b"short"
                )
                path.write_bytes(payload)
                row[key] = str(path.relative_to(bundle))
                row[key + "_sha256"] = core.file_sha(path)
            rows.append(row)
        core.write_jsonl(bundle / "pages.jsonl", rows)
        core.write_json(bundle / "bundle.json", {
            "ready": True, "pages": 2,
            "pages_sha256": core.file_sha(bundle / "pages.jsonl"),
        })
        config = core.read_json(ROOT / "configs/stage1_2250_12k.json")
        config["max_seq_length"] = 8
        core.write_json(root / "config.json", config)
        core.write_json(root / "lock.json", {
            "model_id": core.MODEL_ID, "revision": "a" * 40,
        })
        args = argparse.Namespace(
            bundle=str(bundle), config=str(root / "config.json"),
            model_lock=str(root / "lock.json"), out=str(root / "prepared"),
            arms=["original", "verified"], overflow_policy="common-fit",
            expect_pages=expect, max_overflow_pages=budget,
        )
        tokenizer = types.SimpleNamespace(
            all_special_tokens=[],
            convert_tokens_to_ids=lambda token: 99,
            encode=lambda text, **kwargs: list(text),
        )
        processor = types.SimpleNamespace(
            tokenizer=tokenizer,
            save_pretrained=lambda path: Path(path).mkdir(),
        )
        image = types.SimpleNamespace(close=lambda: None)
        opened = types.SimpleNamespace(convert=lambda mode: image)
        fake_pil = types.ModuleType("PIL")
        fake_pil.Image = types.SimpleNamespace(
            open=lambda path: contextlib.nullcontext(opened)
        )

        def encode(processor, image, text):
            ids = [9, 99]
            if text is not None:
                ids += list(range(10, 18)) if text == "LONG" else [10, 11]
            return {
                "input_ids": [Vector(ids)],
                "image_grid_thw": Vector([[1, 2, 2]]),
            }

        real_copy = preparer.shutil.copyfile

        def copy(source, destination):
            result = real_copy(source, destination)
            if corrupt and Path(destination).name == "original.html":
                Path(destination).write_bytes(b"changed")
            return result

        with contextlib.ExitStack() as stack:
            stack.enter_context(mock.patch.dict(sys.modules, {"PIL": fake_pil}))
            overrides = {
                "verify_model_lock": lambda lock: root,
                "load_processor": lambda *args: processor,
                "conversation_text": lambda processor, html=None: html,
                "encode_image_text": encode,
                "package_versions": lambda: {"fixture": "1"},
            }
            for name, value in overrides.items():
                stack.enter_context(mock.patch.object(preparer, name, value))
            stack.enter_context(mock.patch.object(preparer.shutil, "copyfile", copy))
            preparer.prepare(args)
        return Path(args.out)

    def test_count_mismatch_leaves_no_ready_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(ValueError, "Preflight failed"):
                self.run_fixture(root, expect=3)
            self.assertFalse((root / "prepared/prepared.json").exists())
            issues = core.read_json(root / "prepared/preparation_issues.json")
            self.assertEqual(issues["candidate_retained_pages"], 2)

    def test_common_fit_excludes_one_page_from_both_arms(self):
        with tempfile.TemporaryDirectory() as directory:
            out = self.run_fixture(Path(directory), expect=1, long_page=True, budget=1)
            for arm in ("original", "verified"):
                rows = core.read_jsonl(out / f"{arm}.jsonl")
                self.assertEqual([row["page_id"] for row in rows], ["1"])
            self.assertTrue(core.read_json(out / "prepared.json")["ready"])

    def test_overflow_budget_leaves_no_ready_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(ValueError, "declared budget"):
                self.run_fixture(root, expect=1, long_page=True, budget=0)
            self.assertFalse((root / "prepared/prepared.json").exists())

    def test_changed_copy_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(ValueError, "Preflight failed"):
                self.run_fixture(root, expect=2, corrupt=True)
            self.assertFalse((root / "prepared/prepared.json").exists())


if __name__ == "__main__":
    unittest.main()
