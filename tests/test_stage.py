"""Behavioral checks for time boundaries, exposure inputs and training-only fit."""
import unittest

import numpy as np
import pandas as pd

from rec_lab.data import FeatureEncoder, chronological_split, engineered
from scripts.prepare_stage import label_blind_sample, quality_masks, TIME_MIN


def fixture():
    n = 90
    return pd.DataFrame({'user_id': np.arange(n) % 5, 'video_id': np.arange(n) % 7,
        'tab': np.ones(n, dtype=int), 'time_ms': TIME_MIN + (np.arange(n) // 3) * 3600000,
        'duration_ms': 10000 + np.arange(n), 'is_rand': np.zeros(n, dtype=int),
        'is_click': np.arange(n) % 2, 'is_like': np.arange(n) % 3 == 0,
        'play_time_ms': np.zeros(n), 'source_row_id': [f'fixture:{i}' for i in range(n)]})


class StageChecks(unittest.TestCase):
    def test_zero_auxiliary_weight_ignores_auxiliary_labels(self):
        # Behavioral ablation check: changing only auxiliary labels cannot change
        # the like predictions when that task's training weight is zero.
        import tempfile
        from pathlib import Path
        from rec_lab.data import make_demo, prepare
        from rec_lab.runner import train, saved_probabilities, read_json
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); raw=root/'raw.csv'; make_demo(raw,600)
            prepare([raw],root/'a','synthetic')
            changed=pd.read_csv(raw); changed['is_click']=1-changed.is_click; changed.to_csv(root/'changed.csv',index=False)
            prepare([root/'changed.csv'],root/'b','synthetic')
            for name in ['a','b']:
                train(root/name,root/('run_'+name),model='shared',task='both',aux_weight=0.,
                      selection_target='is_like',epochs=3,seed=42)
                hist=pd.read_csv(root/('run_'+name)/'history.csv')
                cfg=read_json(root/('run_'+name)/'config.json')
                self.assertEqual(cfg['best_epoch'],int(hist.loc[hist.valid_is_like_loss.idxmin(),'epoch']))
            valid=pd.read_csv(root/'a'/'valid.csv')
            pa=saved_probabilities(root/'run_a',valid)[:,1]
            pb=saved_probabilities(root/'run_b',valid)[:,1]
            np.testing.assert_array_equal(pa,pb)

    def test_tied_timestamps_do_not_cross_global_boundaries(self):
        frame = fixture()
        parts = chronological_split(frame, fractions=(.7, .15, .15))
        self.assertLess(parts['train'].time_ms.max(), parts['valid'].time_ms.min())
        self.assertLess(parts['valid'].time_ms.max(), parts['test'].time_ms.min())
        self.assertEqual(sum(map(len, parts.values())), len(frame))
        for name, other in [('train', 'valid'), ('valid', 'test')]:
            self.assertFalse(set(parts[name].time_ms) & set(parts[other].time_ms))

    def test_sampling_ignores_labels_and_is_repeatable(self):
        frame = fixture()
        changed = frame.copy()
        changed['is_click'] = 1 - changed.is_click
        changed['is_like'] = 1 - changed.is_like.astype(int)
        a = label_blind_sample(frame, rows=40)
        b = label_blind_sample(changed, rows=40)
        self.assertEqual(a.source_row_id.tolist(), b.source_row_id.tolist())
        self.assertEqual(a.source_row_id.tolist(), label_blind_sample(frame, rows=40).source_row_id.tolist())

    def test_same_row_and_future_feedback_cannot_change_inputs(self):
        frame = fixture()
        train, valid = frame.iloc[:60].copy(), frame.iloc[60:].copy()
        enc = FeatureEncoder().fit(train)
        before = enc.to_dict()
        expected = enc.encode(valid)
        valid['is_click'] = 1 - valid.is_click
        valid['is_like'] = 1 - valid.is_like.astype(int)
        valid['play_time_ms'] = 9999999
        got = enc.encode(valid)
        np.testing.assert_array_equal(got[0], expected[0])
        np.testing.assert_array_equal(got[1], expected[1])
        self.assertEqual(before, enc.to_dict())

    def test_vocab_and_scaling_only_fit_training_and_unknown_is_zero(self):
        frame = fixture()
        train = frame.iloc[:60].copy()
        valid = frame.iloc[60:].copy()
        valid['user_id'] = 999999
        valid['video_id'] = 888888
        valid['duration_ms'] = 999999
        enc = FeatureEncoder().fit(train)
        cat, _ = enc.encode(valid)
        self.assertTrue((cat[:, :2] == 0).all())
        self.assertNotIn('999999', enc.maps['user_id'])
        np.testing.assert_allclose(enc.mean, engineered(train)[enc.numbers].to_numpy(np.float32).mean(axis=0))
        restored = FeatureEncoder.from_dict(enc.to_dict())
        for a, b in zip(enc.encode(valid), restored.encode(valid)):
            np.testing.assert_array_equal(a, b)

    def test_zero_viewing_and_unclicked_negatives_are_valid(self):
        frame = fixture()
        frame['is_click'] = 0
        frame['is_like'] = 0
        self.assertTrue(all(not mask.any() for mask in quality_masks(frame).values()))
        frame.loc[0, 'duration_ms'] = 0
        self.assertEqual(int(quality_masks(frame)['duration_zero'].sum()), 1)


if __name__ == '__main__':
    unittest.main()
