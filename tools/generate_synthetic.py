"""Generate fictitious integer-only examples, without reading any private data."""
import json
from pathlib import Path
import random

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / 'fixtures' / 'synthetic'


def generate(output=DEFAULT_OUTPUT):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rng = random.Random(2026)
    cardinality = dict(cases=100, symptoms=48, acupoints=24, diseases=8, syndromes=12)
    rows, ordinary = [], []
    for index in range(cardinality['cases']):
        symptoms = sorted(rng.sample(range(48), rng.randint(3, 7)))
        disease = (symptoms[0] + symptoms[1]) % 8
        syndrome = (symptoms[-1] + len(symptoms)) % 12
        # Arbitrary integer relationships: no clinical meaning or real distribution.
        main = sorted({(symptoms[0] * 3 + offset) % 24 for offset in (0, 5, 11)})
        adjunct = sorted({(symptoms[-1] + offset) % 24 for offset in (2, 9)} - set(main))
        if not adjunct:
            adjunct = [next(item for item in range(24) if item not in main)]
        fields = [symptoms, [syndrome], [disease], main, adjunct]
        rows.append('\t'.join(' '.join(map(str, field)) for field in fields))
        ordinary.append(' '.join(map(str, symptoms)) + '\t' + ' '.join(map(str, sorted(set(main) | set(adjunct)))))
    indices = list(range(100))
    random.Random(2026).shuffle(indices)
    folds = []
    for fold in range(5):
        test = sorted(indices[fold * 20:(fold + 1) * 20])
        folds.append({'train': [i for i in range(100) if i not in set(test)], 'test': test})
    manifest = dict(synthetic=True, generator_version=1, seed=2026, cardinality=cardinality)
    # Explicit bytes keep fixtures identical on Windows and Linux, including Python 3.9.
    (output/'r01_cases_5col.txt').write_bytes(('\n'.join(rows)+'\n').encode('utf8'))
    (output/'gcn_cases.txt').write_bytes(('\n'.join(ordinary)+'\n').encode('utf8'))
    for name, value in [('manifest.json', manifest), ('folds.json', folds)]:
        (output/name).write_bytes((json.dumps(value, indent=2)+'\n').encode('utf8'))
    return manifest


if __name__ == '__main__':
    print(json.dumps(generate(), sort_keys=True))
