import unittest
import numpy as np
from rec_lab.uncertainty import paired_user_bootstrap, metrics


class UncertaintyTests(unittest.TestCase):
    def test_identical_predictions_have_exactly_zero_difference(self):
        y=np.array([0,1,0,1,0,1]);users=np.repeat([1,2,3],2)
        p=np.array([[.1,.8,.4,.6,.2,.7],[.2,.7,.3,.7,.3,.8]])
        result,draws=paired_user_bootstrap(y,users,p,p,resamples=20)
        for value in result['metrics'].values():
            self.assertEqual(value['difference'],0)
            self.assertEqual(value['ci95'],[0,0])
        self.assertEqual(len(draws),20)

    def test_weighted_metrics_match_duplicating_user_events(self):
        y=np.array([0,1,0,1]);p=np.array([.1,.9,.7,.6]);w=np.array([2,2,1,1])
        np.testing.assert_allclose(metrics(y,p,w),metrics(np.repeat(y,w),np.repeat(p,w)),atol=1e-12)

    def test_rejects_mismatched_shapes_and_invalid_probabilities(self):
        for a,b in [(np.array([[.1,.9]]),np.array([[.2]])),(np.array([[.1,1.9]]),np.array([[.2,.8]]))]:
            with self.assertRaises(ValueError):paired_user_bootstrap([0,1],[1,2],a,b,resamples=10)


if __name__=='__main__':unittest.main()
