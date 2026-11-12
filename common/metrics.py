"""Standard binary-relevance ranking metrics for the synthetic demonstration."""
import numpy as np


def ranking_metrics(scores, labels, ks):
    scores = np.asarray(scores)
    if scores.ndim != 2 or scores.shape[0] != len(labels) or not np.isfinite(scores).all():
        raise ValueError('Expected finite case-by-candidate score matrix')
    if not len(labels):
        raise ValueError('Cannot evaluate an empty dataset')
    n_items = scores.shape[1]
    result = {}
    for k in ks:
        if not 0 < k <= n_items:
            raise ValueError('K must not exceed candidate count')
        precision, recall, ndcg = [], [], []
        discounts = 1 / np.log2(np.arange(k) + 2.)
        for row, truth in zip(scores, labels):
            truth = set(truth)
            if any(not 0 <= item < n_items for item in truth):
                raise ValueError('Label outside candidate range')
            order = np.lexsort((np.arange(n_items), -row))[:k]
            hit = np.array([int(item in truth) for item in order])
            precision.append(float(hit.sum()) / k)
            if truth:
                recall.append(float(hit.sum()) / len(truth))
                ideal = discounts[:min(k, len(truth))].sum()
                ndcg.append(float((hit * discounts).sum() / ideal))
        result[str(k)] = dict(recall=float(np.mean(recall)) if recall else None,
                              precision=float(np.mean(precision)),
                              ndcg=float(np.mean(ndcg)) if ndcg else None)
    return result
