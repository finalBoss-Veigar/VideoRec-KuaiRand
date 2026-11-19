import json
import shutil
import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from rec_lab.data import make_demo,prepare,chronological_split,FeatureEncoder
from rec_lab.runner import train,evaluate_saved,predict_saved,load_parts
from rec_lab.metrics import binary_metrics
from rec_lab.experiments import suite
from rec_lab.report import report

class V2Checks(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        self.raw=self.root/'raw.csv';make_demo(self.raw,rows=600)
    def tearDown(self):self.tmp.cleanup()
    def test_output_preserved(self):
        out=self.root/'data';prepare([self.raw],out,'synthetic')
        original=(out/'manifest.json').read_bytes()
        with self.assertRaisesRegex(ValueError,'already exists'):prepare([self.raw],out,'synthetic')
        self.assertEqual(original,(out/'manifest.json').read_bytes())
    def test_invalid_row_decision_recorded(self):
        d=pd.read_csv(self.raw);d.loc[0,'duration_ms']=0;d.to_csv(self.raw,index=False)
        with self.assertRaisesRegex(ValueError,'Invalid rows'):prepare([self.raw],self.root/'bad','synthetic')
        self.assertFalse((self.root/'bad').exists())
        m=prepare([self.raw],self.root/'good','synthetic',invalid_rows='drop')
        self.assertEqual(m['dropped_invalid_rows'],1);self.assertEqual(sum(x['rows'] for x in m['split_audit'].values()),599)
    def test_fractional_id_rejected(self):
        d=pd.read_csv(self.raw);d['user_id']=d.user_id.astype(float);d.loc[0,'user_id']=1.5;d.to_csv(self.raw,index=False)
        with self.assertRaises(ValueError):prepare([self.raw],self.root/'bad','synthetic')
    def test_negative_subset_rejected(self):
        with self.assertRaises(ValueError):prepare([self.raw],self.root/'bad','synthetic',max_rows=-1)
    def test_explicit_cutoffs(self):
        d=pd.read_csv(self.raw);t1=pd.to_datetime(d.time_ms.iloc[400],unit='ms',utc=True);t2=pd.to_datetime(d.time_ms.iloc[500],unit='ms',utc=True)
        parts=chronological_split(d,cutoffs=(str(t1),str(t2)))
        self.assertEqual([len(parts[x]) for x in ['train','valid','test']],[400,100,100])
    def test_manifest_change_rejected(self):
        out=self.root/'data';prepare([self.raw],out,'synthetic')
        m=json.loads((out/'manifest.json').read_text(encoding='utf-8'));m['signal_note']='changed';(out/'manifest.json').write_text(json.dumps(m),encoding='utf-8')
        with self.assertRaisesRegex(ValueError,'protocol changed'):load_parts(out)
    def test_unlabeled_input_and_numeric_id_normalization(self):
        d=pd.read_csv(self.raw);enc=FeatureEncoder().fit(d)
        a,b=enc.encode(d.drop(columns=['is_click','is_like']))
        changed=d.copy();changed.user_id=changed.user_id.astype(float)
        a2,b2=enc.encode(changed)
        np.testing.assert_array_equal(a,a2);np.testing.assert_allclose(b,b2)
    def test_empty_prediction_fails(self):
        d=pd.read_csv(self.raw);enc=FeatureEncoder().fit(d)
        with self.assertRaises(ValueError):enc.encode(d.iloc[:0])
    def test_bad_probabilities_and_labels_fail(self):
        for y,p in [([0,1],[.1,1.1]),([0,.5],[.1,.2]),([],[])]:
            with self.assertRaises(ValueError):binary_metrics(y,p)
    def test_portable_model_and_test_guard(self):
        bundle=self.root/'old';data=bundle/'data';run=bundle/'runs/model'
        prepare([self.raw],data,'synthetic');train(data,run,model='constant')
        self.assertNotIn('test',json.loads((run/'metrics.json').read_text(encoding='utf-8')))
        relocated=self.root/'moved';shutil.move(str(bundle),relocated)
        run=relocated/'runs/model';evaluate_saved(run)
        before=(run/'metrics.json').read_bytes()
        with self.assertRaisesRegex(ValueError,'already evaluated'):evaluate_saved(run)
        self.assertEqual(before,(run/'metrics.json').read_bytes())
        df=pd.read_csv(relocated/'data/test.csv').drop(columns=['is_click','is_like']);df.to_csv(self.root/'unlabeled.csv',index=False)
        predict_saved(run,self.root/'unlabeled.csv',self.root/'pred.csv')
        self.assertIn('p_is_click',pd.read_csv(self.root/'pred.csv'))
    def test_bad_suite_rejected_before_training(self):
        data=self.root/'data';prepare([self.raw],data,'synthetic')
        cfg=self.root/'bad.yaml';cfg.write_text('data: '+str(data)+'\noutput: '+str(self.root/'runs')+'\nexperiments:\n  - {name: x, model: lr, typo: 1}\n')
        with self.assertRaisesRegex(ValueError,'Unknown experiment'):suite(cfg)
        self.assertFalse((self.root/'runs').exists())
    def test_suite_skip_summary_and_changed_request(self):
        data=self.root/'data';prepare([self.raw],data,'synthetic')
        cfg=self.root/'suite.yaml';out=self.root/'runs'
        text='data: '+str(data)+'\noutput: '+str(out)+'\nseeds: [42,43]\nexperiments:\n  - {name: c, model: constant, task: engage}\n'
        cfg.write_text(text);suite(cfg)
        weights=(out/'c_s42/constant.json').read_bytes()
        suite(cfg)
        self.assertEqual(weights,(out/'c_s42/constant.json').read_bytes())
        summary=pd.read_csv(out/'seed_summary.csv');self.assertEqual(int(summary.auc_count.iloc[0]),2)
        cfg.write_text(text.replace('model: constant','model: lr'))
        with self.assertRaisesRegex(ValueError,'different request'):suite(cfg)
    def test_report_is_self_contained_and_has_calibration(self):
        data=self.root/'data';prepare([self.raw],data,'synthetic');run=self.root/'run'
        train(data,run,model='constant')
        result=report([run],self.root/'report')
        page=Path(result['report']).read_text(encoding='utf-8')
        self.assertIn('data:image/png;base64,',page)
        self.assertIn('synthetic',page)
        self.assertTrue((self.root/'report/groups.csv').exists())
    def test_single_class_group_has_positive_count(self):
        m=binary_metrics([1,1],[.8,.7]);self.assertEqual(m['positive_count'],2)
        self.assertIsNone(m['auc'])
    def test_custom_architecture_survives_unlabeled_prediction(self):
        data=self.root/'data';prepare([self.raw],data,'synthetic');run=self.root/'custom'
        train(data,run,model='mmoe',task='both',epochs=2,embedding_dim=5,hidden=(12,7),experts=2,variant='no_user')
        df=pd.read_csv(data/'valid.csv').drop(columns=['is_click','is_like']);df.to_csv(self.root/'input.csv',index=False)
        predict_saved(run,self.root/'input.csv',self.root/'custom.csv')
        predictions=pd.read_csv(self.root/'custom.csv')
        self.assertTrue(predictions.p_is_click.between(0,1).all())
        self.assertTrue(predictions.p_is_like.between(0,1).all())

if __name__=='__main__':unittest.main()
