"""Paired primary contrasts from validated case-level revision scores."""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.stats import binomtest
from .revision_core import paired_rows, read_jsonl, write_json


def holm(pvalues):
    order = np.argsort(pvalues)
    adjusted = [0.0] * len(pvalues)
    previous = 0.0
    for rank, index in enumerate(order):
        previous = max(previous, min(1.0, (len(order) - rank) * pvalues[index]))
        adjusted[index] = previous
    return adjusted


def analyze(rows, metric, clusters, seed=3407, samples=10000, adjust_interaction=True):
    ids, values = paired_rows(rows, metric)
    if any(case not in clusters for case in ids):
        raise ValueError('Missing test cluster IDs')
    values = np.asarray(values, dtype=float)
    groups = [[i for i, case in enumerate(ids) if clusters[case] == group]
              for group in sorted({clusters[case] for case in ids})]
    rng = np.random.default_rng(seed)
    weights = np.asarray([[-1, 1, 0, 0], [-1, 0, 1, 0], [-1, 0, 0, 1], [1, -1, -1, 1]])
    names = ('rag_minus_base', 'lora_minus_base', 'combined_minus_base', 'interaction')
    deltas = values @ weights.T
    bootstrap = np.empty((samples, 4))
    bootstrap_means = np.empty((samples, 4))
    null = np.empty((samples, 4))
    group_sums = np.asarray([deltas[g].sum(axis=0) for g in groups])
    for i in range(samples):
        selected = rng.integers(len(groups), size=len(groups))
        indices = np.concatenate([groups[g] for g in selected])
        bootstrap[i] = deltas[indices].mean(axis=0)
        bootstrap_means[i] = values[indices].mean(axis=0)
        null[i] = (group_sums * rng.choice([-1, 1], size=(len(groups), 1))).sum(axis=0) / len(ids)
    binary = bool(np.isin(values, [0, 1]).all())
    contrasts = []
    for j, name in enumerate(names):
        observed = float(deltas[:, j].mean())
        p = float((1 + (np.abs(null[:, j]) >= abs(observed) - 1e-12).sum()) / (samples + 1))
        method = 'cluster sign-flip permutation'
        # Exact McNemar requires independent pairs; duplicate clusters use sign flips.
        if binary and j < 3 and len(groups) == len(ids):
            target = (1, 2, 3)[j]
            positive = int(((values[:, target] == 1) & (values[:, 0] == 0)).sum())
            negative = int(((values[:, target] == 0) & (values[:, 0] == 1)).sum())
            p = float(binomtest(positive, positive + negative).pvalue) if positive + negative else 1.0
            method = 'exact McNemar'
        contrasts.append(dict(contrast=name, difference=observed,
                              ci95=np.quantile(bootstrap[:, j], [.025, .975]).tolist(), p=p, test=method))
    family = contrasts if adjust_interaction else contrasts[:3]
    for result, adjusted in zip(family, holm([r['p'] for r in family])):
        result['p_holm'] = adjusted
    if not adjust_interaction:
        contrasts[3]['p_holm'] = None
        contrasts[3]['analysis_role'] = 'exploratory interaction; unadjusted p only'
    return dict(metric=metric, n_cases=len(ids), n_clusters=len(groups), seed=seed,
                resamples=samples, means=dict(zip(('base', 'rag', 'lora', 'lora_rag'), values.mean(axis=0).tolist())),
                condition_ci95=dict(zip(('base', 'rag', 'lora', 'lora_rag'), np.quantile(bootstrap_means, [.025, .975], axis=0).T.tolist())),
                success_counts=values.sum(axis=0).astype(int).tolist() if binary else None,
                holm_family=[r['contrast'] for r in family], contrasts=contrasts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scores', type=Path, required=True)
    parser.add_argument('--audit', type=Path, required=True)
    parser.add_argument('--metric', default='faithfulness')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    audit = json.loads(args.audit.read_text(encoding='utf-8'))
    rows = read_jsonl(args.scores)
    ids, _ = paired_rows(rows, args.metric)
    if set(ids) != set(audit['test_clusters']):
        raise ValueError('Scores do not cover the complete audited test cohort')
    write_json(args.output, analyze(rows, args.metric, audit['test_clusters']))


if __name__ == '__main__':
    main()
