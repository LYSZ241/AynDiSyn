"""Strict loading and alignment of integer-only case datasets."""
import json
from pathlib import Path
import random
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT/'datasets'/'numeric'


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_ids(text, size):
    tokens = text.split()
    if any(not token.isascii() or not token.isdecimal() for token in tokens):
        raise ValueError('Only nonnegative integer IDs are accepted')
    result = [int(token) for token in tokens]
    if len(result) != len(set(result)) or any(not 0 <= value < size for value in result):
        raise ValueError('Duplicate or out-of-range ID')
    return result


class Data:
    def __init__(self, path=DEFAULT_DATA, fold=0):
        self.path = Path(path)
        self.manifest = json.loads((self.path/'manifest.json').read_text(encoding='utf8'))
        if type(self.manifest.get('synthetic')) is not bool:
            raise ValueError('Manifest must explicitly declare synthetic as true or false')
        self.synthetic = self.manifest['synthetic']
        counts = self.manifest['cardinality']
        self.n_users, self.n_items = counts['symptoms'], counts['acupoints']
        lines = (self.path/'r01_cases_5col.txt').read_text(encoding='utf8').splitlines()
        ordinary = (self.path/'gcn_cases.txt').read_text(encoding='utf8').splitlines()
        if len(lines) != counts['cases'] or len(ordinary) != len(lines):
            raise ValueError('Case counts do not match manifest')
        self.cases = []
        for row, raw in enumerate(lines):
            fields = raw.split('\t')
            if len(fields) != 5:
                raise ValueError('R01 requires five tab-separated columns')
            sizes = (self.n_users, counts['syndromes'], counts['diseases'], self.n_items, self.n_items)
            values = [parse_ids(text, size) for text, size in zip(fields, sizes)]
            symptoms, syndromes, diseases, main, adjunct = values
            if not symptoms or not (main or adjunct):
                raise ValueError('Rows require symptoms and at least one acupoint across both roles')
            two = ordinary[row].split('\t')
            if len(two) != 2:
                raise ValueError('GCN requires two tab-separated columns')
            acupoints = parse_ids(two[1], self.n_items)
            if parse_ids(two[0], self.n_users) != symptoms or set(acupoints) != set(main) | set(adjunct):
                raise ValueError('GCN and R01 rows do not align')
            self.cases.append(dict(symptom_ids=symptoms, syndrome_ids=syndromes, disease_ids=diseases,
                                   main_ids=main, adjunct_ids=adjunct, acupoint_ids=acupoints))
        folds = json.loads((self.path/'folds.json').read_text(encoding='utf8'))
        if len(folds) != 5 or not 0 <= fold < 5:
            raise ValueError('Expected five folds and fold index 0..4')
        seen_test = []
        expected = set(range(len(self.cases)))
        for split in folds:
            train, test = split['train'], split['test']
            if (len(train) != len(set(train)) or len(test) != len(set(test))
                    or set(train) & set(test) or set(train) | set(test) != expected):
                raise ValueError('Invalid train/test split')
            seen_test.extend(test)
        if sorted(seen_test) != sorted(expected):
            raise ValueError('Each case must appear in exactly one test fold')
        self.split = folds[fold]
        self.train = [self.cases[i] for i in self.split['train']]
        self.test = [self.cases[i] for i in self.split['test']]


def batch_tensors(cases, n_items, device):
    length = max(len(case['symptom_ids']) for case in cases)
    ids = torch.zeros((len(cases), length), dtype=torch.long, device=device)
    mask = torch.zeros((len(cases), length), dtype=torch.float32, device=device)
    targets = {name: torch.zeros((len(cases), n_items), device=device)
               for name in ('union', 'main', 'adjunct')}
    for index, case in enumerate(cases):
        values = case['symptom_ids']
        ids[index, :len(values)] = torch.tensor(values, device=device)
        mask[index, :len(values)] = 1
        for name, key in [('union', 'acupoint_ids'), ('main', 'main_ids'), ('adjunct', 'adjunct_ids')]:
            targets[name][index, case[key]] = 1
    return ids, mask, targets
