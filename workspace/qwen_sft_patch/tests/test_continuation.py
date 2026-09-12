import tempfile
import unittest
from pathlib import Path
from sft_core import *

class ContinuationTests(unittest.TestCase):
    def fixture(self, root):
        c=load_config(Path(__file__).resolve().parents[1]/'configs/stage1_2000.json')
        lock={'model_id':MODEL_ID,'revision':'a'*40}
        recipe={'training_method':'bf16_lora','arm':'original','model_id':MODEL_ID,
                'model_revision':lock['revision'],'config':c,'training_page_ids':['old'],
                'ancestor_page_ids':[], 'data_fingerprint':'paired-data','code_hashes':{}}
        folder=root/'final_adapter';folder.mkdir()
        (folder/'adapter_model.safetensors').write_bytes(b'fixture; no tensor parsing in this CPU integrity test')
        write_json(folder/'adapter_config.json',{})
        write_json(folder/'training_recipe.json',recipe)
        h=digest(recipe)
        write_json(root/'recipe.json', recipe|{'recipe_hash':h})
        write_json(root/'summary.json',{'status':'completed','smoke':False,'arm':'original','recipe_hash':h})
        write_json(root/'final_adapter_hashes.json',{p.name:file_sha(p) for p in folder.iterdir()})
        return c,lock
    def test_valid_parent_and_disjoint_pages(self):
        with tempfile.TemporaryDirectory() as d:
            c,l=self.fixture(Path(d));c['experiment']='stage2'
            p=validate_parent_run(d,'original',c,l,['new'])
            self.assertEqual(p['ancestor_page_ids'],['old'])
    def test_wrong_arm_and_overlap_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            c,l=self.fixture(Path(d))
            for arm,ids in [('verified',['new']),('original',['old'])]:
                with self.assertRaises(ValueError):validate_parent_run(d,arm,c,l,ids)
    def test_model_config_and_corruption_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);c,l=self.fixture(root)
            with self.assertRaisesRegex(ValueError,'revision'):validate_parent_run(d,'original',c,l|{'revision':'b'*40},['new'])
            with self.assertRaisesRegex(ValueError,'lora_r'):validate_parent_run(d,'original',c|{'lora_r':16},l,['new'])
            (root/'final_adapter/adapter_model.safetensors').write_bytes(b'corrupt')
            with self.assertRaisesRegex(ValueError,'changed'):validate_parent_run(d,'original',c,l,['new'])
