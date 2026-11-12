"""Synthetic-only checks: no private files or network services are accessed."""
import math
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch

from common.data import Data, batch_tensors
from common.metrics import ranking_metrics
from gcn.gcn_model3 import DemoModel as SingleDemo
from R01.gcn_model3 import DemoModel as DualDemo
from tools.generate_synthetic import DEFAULT_OUTPUT, generate


class PublicDemoTests(unittest.TestCase):
    def test_fixture_is_regenerated_exactly_without_external_inputs(self):
        with tempfile.TemporaryDirectory() as folder:
            generate(folder)
            for filename in ('manifest.json', 'folds.json', 'gcn_cases.txt', 'r01_cases_5col.txt'):
                self.assertEqual((Path(folder)/filename).read_bytes(), (DEFAULT_OUTPUT/filename).read_bytes())

    def test_all_rows_align_and_five_test_folds_cover_once(self):
        tested = []
        for fold in range(5):
            data = Data(fold=fold)
            self.assertEqual((len(data.train), len(data.test)), (80, 20))
            tested.extend(data.split['test'])
            for case in data.cases:
                self.assertEqual(set(case['acupoint_ids']), set(case['main_ids']) | set(case['adjunct_ids']))
        self.assertEqual(sorted(tested), list(range(100)))

    def test_metric_definition_tie_order_and_empty_ground_truth(self):
        scores = np.array([[4., 4., 2., 1.], [4., 3., 2., 1.]])
        values = ranking_metrics(scores, [[0, 2], []], [1, 2])
        self.assertEqual(values['1']['precision'], .5)
        self.assertEqual(values['2']['precision'], .25)
        self.assertEqual(values['2']['recall'], .5)
        self.assertAlmostEqual(values['2']['ndcg'], 1/(1+1/math.log2(3)), places=12)
        with self.assertRaises(ValueError):
            ranking_metrics(scores, [[0], []], [5])

    def test_padding_does_not_change_mean_or_scores(self):
        model = SingleDemo(48, 24, 8).eval()
        ids = torch.tensor([[1, 2]])
        padded = torch.tensor([[1, 2, 47, 46]])
        a = model(ids, torch.tensor([[1., 1.]]))['union']
        b = model(padded, torch.tensor([[1., 1., 0., 0.]]))['union']
        torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_both_demo_heads_receive_training_gradients(self):
        data = Data(fold=0)
        ids, mask, targets = batch_tensors(data.train[:4], data.n_items, 'cpu')
        model = DualDemo(data.n_users, data.n_items, 8)
        scores = model(ids, mask)
        loss_fn = torch.nn.BCEWithLogitsLoss()
        loss = sum(loss_fn(value, targets[key]) for key, value in scores.items())
        loss.backward()
        for name in ('embedding', 'main_output', 'adjunct_output'):
            gradient = getattr(model, name).weight.grad
            self.assertTrue(torch.isfinite(gradient).all())
            self.assertGreater(float(gradient.norm()), 0.)


if __name__ == '__main__':
    unittest.main()
