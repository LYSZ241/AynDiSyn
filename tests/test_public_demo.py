"""Synthetic-only checks: no private files or network services are accessed."""
import math
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch

from common.data import Data, batch_tensors
from common.metrics import ranking_metrics
from common.training import display, evaluate
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
            data = Data(DEFAULT_OUTPUT, fold=fold)
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
        data = Data(DEFAULT_OUTPUT, fold=0)
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

    def test_numeric_schema_keeps_empty_fields_and_overlapping_roles(self):
        # All content is generated in this test: no real dataset is accessed.
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            generate(folder)
            manifest = json.loads((folder/'manifest.json').read_text(encoding='utf8'))
            manifest['synthetic'] = False
            (folder/'manifest.json').write_text(json.dumps(manifest), encoding='utf8')
            rows = (folder/'r01_cases_5col.txt').read_text(encoding='utf8').splitlines()
            ordinary = (folder/'gcn_cases.txt').read_text(encoding='utf8').splitlines()
            first = rows[0].split('\t')
            first[1], first[2], first[4] = '', '', ''
            rows[0] = '\t'.join(first)
            reversed_acupoints = list(reversed(first[3].split()))
            ordinary[0] = first[0] + '\t' + ' '.join(reversed_acupoints)
            second = rows[1].split('\t')
            overlap = second[3].split()[0]
            second[4] = overlap + ' ' + second[4]
            rows[1] = '\t'.join(second)
            (folder/'r01_cases_5col.txt').write_text('\n'.join(rows)+'\n', encoding='utf8')
            (folder/'gcn_cases.txt').write_text('\n'.join(ordinary)+'\n', encoding='utf8')
            data = Data(folder, fold=0)
            self.assertFalse(data.synthetic)
            case = data.cases[0]
            self.assertEqual(case['syndrome_ids'], [])
            self.assertEqual(case['disease_ids'], [])
            self.assertEqual(case['adjunct_ids'], [])
            self.assertEqual(case['main_ids'], [int(x) for x in first[3].split()])
            self.assertEqual(case['acupoint_ids'], [int(x) for x in reversed_acupoints])
            self.assertIn(int(overlap), data.cases[1]['main_ids'])
            self.assertIn(int(overlap), data.cases[1]['adjunct_ids'])
            _, _, targets = batch_tensors(data.cases[:2], data.n_items, 'cpu')
            self.assertEqual(float(targets['adjunct'][0].sum()), 0.)
            self.assertEqual(float(targets['main'][1, int(overlap)]), 1.)
            self.assertEqual(float(targets['adjunct'][1, int(overlap)]), 1.)
            model = DualDemo(data.n_users, data.n_items, 8)
            metrics = evaluate(model, [case], data.n_items, 'cpu', [5], 1, 'R01')
            self.assertIsNone(metrics['adjunct']['5']['recall'])
            self.assertIsNone(metrics['adjunct']['5']['ndcg'])
            self.assertEqual(metrics['adjunct']['5']['precision'], 0.)
            with contextlib.redirect_stdout(io.StringIO()) as output:
                display('test', metrics, [5])
            self.assertIn('N/A', output.getvalue())


if __name__ == '__main__':
    unittest.main()
