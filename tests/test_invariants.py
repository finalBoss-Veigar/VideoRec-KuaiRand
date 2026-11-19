import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from rec_lab.data import make_demo,chronological_split,FeatureEncoder,prepare
from rec_lab.metrics import binary_metrics
from rec_lab.models import fm_interaction,build_model

class Invariants(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.root=Path(self.tmp.name)
        make_demo(self.root/'demo.csv',rows=600)
        self.df=pd.read_csv(self.root/'demo.csv')
    def tearDown(self):
        self.tmp.cleanup()
    def test_time_ties_stay_together(self):
        df=self.df.copy();df.time_ms=np.repeat(np.arange(200),3)
        p=chronological_split(df.sample(frac=1,random_state=7))
        self.assertEqual(sum(len(x) for x in p.values()),len(df))
        self.assertLess(p['train'].time_ms.max(),p['valid'].time_ms.min())
        self.assertLess(p['valid'].time_ms.max(),p['test'].time_ms.min())
    def test_unseen_categories_and_scaling(self):
        train=self.df.iloc[:400].copy();enc=FeatureEncoder().fit(train)
        before=enc.to_dict()
        test=self.df.iloc[400:].copy();test.user_id=999999;test.duration_ms=999999999
        c,n,_=enc.transform(test)
        self.assertTrue((c[:,0]==0).all())
        self.assertTrue(np.isfinite(n).all())
        self.assertEqual(enc.to_dict(),before)
    def test_feedback_not_a_feature(self):
        enc=FeatureEncoder().fit(self.df)
        c,n,_=enc.transform(self.df)
        changed=self.df.copy();changed.is_click=1-changed.is_click;changed.is_like=1-changed.is_like
        changed['play_time_ms']=99999999
        c2,n2,_=enc.transform(changed)
        np.testing.assert_array_equal(c,c2);np.testing.assert_array_equal(n,n2)
    def test_fm_matches_explicit_pairs_and_gradient(self):
        x=torch.randn(7,4,3,requires_grad=True)
        direct=sum((x[:,i]*x[:,j]).sum(dim=1) for i in range(4) for j in range(i+1,4))
        torch.testing.assert_close(fm_interaction(x),direct)
        fm_interaction(x).sum().backward()
        self.assertTrue(torch.isfinite(x.grad).all())
    def test_models_shape_gradient_and_reload(self):
        enc=FeatureEncoder().fit(self.df);c,n,_=enc.transform(self.df.iloc[:8])
        c,n=torch.tensor(c),torch.tensor(n)
        for name in ['deepfm','single','shared','mmoe']:
            with self.subTest(model=name):
                net=build_model(name,enc.sizes,3)
                logits=net(c,n)
                self.assertEqual(logits.shape,(8,1 if name in ['deepfm','single'] else 2))
                loss=torch.nn.functional.binary_cross_entropy_with_logits(logits,torch.zeros_like(logits))
                loss.backward()
                self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in net.parameters()))
                torch.save(net.state_dict(),self.root/'model.pt')
                restored=build_model(name,enc.sizes,3)
                restored.load_state_dict(torch.load(self.root/'model.pt',weights_only=True))
                torch.testing.assert_close(net(c,n),restored(c,n))
    def test_real_requires_signal_declaration(self):
        with self.assertRaises(ValueError):
            prepare([self.root/'demo.csv'],self.root/'out','real')
    def test_duplicate_files_rejected(self):
        with self.assertRaisesRegex(ValueError,'Duplicate input'):
            prepare([self.root/'demo.csv',self.root/'demo.csv'],self.root/'out','synthetic')
        self.assertFalse((self.root/'out').exists())
    def test_single_class_metric_is_explicit(self):
        m=binary_metrics([0,0,0],[.1,.2,.3])
        self.assertIsNone(m['auc']);self.assertIsNone(m['pr_auc_average_precision'])
        self.assertTrue(np.isfinite(m['logloss']))

if __name__=='__main__':
    unittest.main()
