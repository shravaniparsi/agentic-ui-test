"""Regression checks for unsafe reference-set release paths."""
import ast, base64, copy, hashlib, importlib.util, json, tempfile, unittest
from typing import Union, Optional
from pathlib import Path
ROOT=Path(__file__).resolve().parent
def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
prep=module('prep',ROOT.parents[1]/'scripts/prepare_validated_visual_dataset.py')
exp=module('exporter',ROOT/'export_approved.py')

class GateTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.p=Path(self.tmp.name)
        self.image=self.p/'ref.jpg';self.image.write_bytes(b'original reference bytes')
        self.entry=dict(target_instance_id='vwa_shopping_1',source_instance_id='vwa_shopping_1',status='approved',identity_verified=True,completion_supported=True,identity_evidence='Exact item',completion_evidence='Confirmed in cart',reviewer='Fixture reviewer',archive_sha256='a'*64,frame_sha256=hashlib.sha256(self.image.read_bytes()).hexdigest(),selection_rationale='Recorded confirmation',reference_screenshot_path=str(self.image))
        self.dataset=self.p/'dataset.jsonl';self.dataset.write_text(json.dumps(dict(instance_id='vwa_shopping_1',task_text='Task',reference_screenshot_path='old.jpg'))+'\n'+json.dumps(dict(instance_id='vwa_shopping_2',task_text='Excluded'))+'\n')
        self.manifest=self.p/'manifest.json';self.output=self.p/'out.jsonl'
    def tearDown(self):self.tmp.cleanup()
    def run_prepare(self,entries=None):
        self.manifest.write_text(json.dumps(entries if entries is not None else [self.entry]))
        return prep.prepare(self.dataset,self.manifest,self.output)
    def test_subset_and_original_preserved(self):
        before=self.dataset.read_bytes();self.assertEqual(self.run_prepare(),1)
        row=json.loads(self.output.read_text());self.assertEqual(row['instance_id'],'vwa_shopping_1');self.assertEqual(row['reference_screenshot_path'],str(self.image.resolve()));self.assertEqual(self.dataset.read_bytes(),before)
    def test_tampered_bytes_block_before_output(self):
        self.image.write_bytes(b'altered')
        with self.assertRaisesRegex(ValueError,'hash mismatch'):self.run_prepare()
        self.assertFalse(self.output.exists())
    def test_unapproved_blocked(self):
        self.entry['status']='unresolved'
        with self.assertRaisesRegex(ValueError,'Unapproved'):self.run_prepare()
        self.assertFalse(self.output.exists())
    def test_task_remapping_blocked(self):
        self.entry['source_instance_id']='vwa_shopping_2'
        with self.assertRaisesRegex(ValueError,'remapping'):self.run_prepare()
    def test_duplicate_and_missing_ids_blocked(self):
        with self.assertRaisesRegex(ValueError,'Duplicate'):self.run_prepare([self.entry,self.entry])
        other=copy.deepcopy(self.entry);other['source_instance_id']=other['target_instance_id']='vwa_shopping_999'
        with self.assertRaisesRegex(ValueError,'missing'):self.run_prepare([other])
    def test_existing_output_preserved(self):
        self.output.write_text('preserve me')
        with self.assertRaises(FileExistsError):self.run_prepare()
        self.assertEqual(self.output.read_text(),'preserve me')
    def test_export_validates_whole_batch_first(self):
        entries=json.loads((ROOT/'review_decisions.json').read_text());good=next(d for d in entries if d['status']=='approved');bad=copy.deepcopy(good);bad['target_instance_id']='vwa_shopping_999';bad['source_instance_id']='../escape'
        path=self.p/'decisions.json';path.write_text(json.dumps([good,bad]));dest=self.p/'export'
        with self.assertRaisesRegex(ValueError,'Invalid source'):exp.export(path,dest)
        self.assertFalse(dest.exists())
    def test_stale_export_blocked(self):
        dest=self.p/'export';dest.mkdir();(dest/'stale.jpg').write_bytes(b'stale')
        with self.assertRaisesRegex(ValueError,'stale'):exp.export(ROOT/'review_decisions.json',dest)
        self.assertEqual(list(dest.iterdir()),[dest/'stale.jpg'])
    def test_mixed_image_mime_and_order_without_sdk_or_credentials(self):
        # Compile only pure message-building functions; never load .env or initialize clients.
        source=ast.parse((ROOT.parents[1]/'llm_clients.py').read_text())
        functions=[n for n in source.body if isinstance(n,ast.FunctionDef) and n.name in {'_encode_image','_detect_media_type','_build_multimodal_messages'}]
        ns=dict(Path=Path,Union=Union,Optional=Optional,base64=base64)
        exec(compile(ast.Module(body=functions,type_ignores=[]),'message_builder','exec'),ns)
        actual=self.p/'actual.png';actual.write_bytes(b'agent PNG fixture')
        messages=ns['_build_multimodal_messages']('system','task',actual,self.image)
        blocks=messages[1]['content']
        self.assertTrue(blocks[0]['image_url']['url'].startswith('data:image/jpeg;base64,'))
        self.assertTrue(blocks[1]['image_url']['url'].startswith('data:image/png;base64,'))
        self.assertEqual(base64.b64decode(blocks[0]['image_url']['url'].split(',')[1]),self.image.read_bytes())
        self.assertEqual(base64.b64decode(blocks[1]['image_url']['url'].split(',')[1]),actual.read_bytes())

if __name__=='__main__':unittest.main(verbosity=2)
