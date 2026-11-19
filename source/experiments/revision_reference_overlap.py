"""Reference overlap versus source support, and source support restricted to structurally valid notes.

Post hoc analysis requested in review. For both held-out datasets it computes, per generated note:
ROUGE-1/2/L F1 against the dataset reference note (rouge_score, Porter stemming), and compares each
intervention arm with Base by the paired mean difference (case bootstrap, 10,000 resamples), the
Wilcoxon signed-rank test and Holm adjustment across the three primary contrasts. It also repeats the
source-support contrasts of every judge on the cases where both arms produced a structurally valid note.
Judges always receive the raw model output, valid JSON or not; empty outputs score 0.

    python -m source.experiments.revision_reference_overlap
Output: revision/results/reference_overlap_v1.json
"""
import json
from pathlib import Path

import numpy as np
from rouge_score import rouge_scorer
from scipy.stats import binomtest, wilcoxon

R = Path('revision/results')
ARMS = ['base', 'rag', 'lora', 'lora_rag', 'random_rag', 'schema']
PRIMARY = ['rag', 'lora', 'lora_rag']
DATASETS = {
    'main': dict(refs=R / 'data_v1/test.jsonl',
                 judges={'gpt-4o-mini': R / 'rescore_v1/scored/{a}.jsonl',
                         'gpt-5.6-luna': R / 'rescore_luna_v1/scored/{a}.jsonl',
                         'Jev v2': R / 'jev_v2/{a}.jsonl'}),
    'aci': dict(refs=R / 'data_aci_v1/test.jsonl',
                judges={'gpt-4o-mini': R / 'aci/gpt4omini_{a}.jsonl',
                        'gpt-5.6-luna': R / 'aci/luna_{a}.jsonl',
                        'Jev v2': R / 'aci/jevv2_{a}.jsonl'}),
}
SEED = 3407


def read(path):
    with open(path, encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def flat(output):
    """Note text for overlap scoring: the four SOAP fields in order, or the raw text if not JSON."""
    try:
        note = json.loads(output)
        if isinstance(note, dict):
            keys = {k.lower(): k for k in note}
            parts = [f'{s.title()}:\n{note[keys[s]]}' for s in ('subjective', 'objective', 'assessment', 'plan')
                     if s in keys]
            if parts:
                return '\n'.join(str(p) for p in parts)
    except (ValueError, TypeError):
        pass
    return str(output or '')


def holm(ps):
    order = np.argsort(ps); m = len(ps); adj = [0.0] * m; run = 0.0
    for i, k in enumerate(order):
        run = max(run, min(1.0, (m - i) * ps[k])); adj[k] = run
    return adj


def boot(diff, rng, n=10000):
    d = np.asarray(diff); idx = rng.integers(0, len(d), (n, len(d)))
    return [float(np.percentile(d[idx].mean(1), 2.5)), float(np.percentile(d[idx].mean(1), 97.5))]


def mcnemar(a, b):
    """Exact McNemar on paired binary outcomes (b vs a); returns discordant counts and p."""
    up = sum(1 for x, y in zip(a, b) if x == 0 and y == 1); down = sum(1 for x, y in zip(a, b) if x == 1 and y == 0)
    p = 1.0 if up + down == 0 else binomtest(up, up + down, 0.5).pvalue
    return up, down, float(p)


def main():
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    out = {'description': __doc__.strip().splitlines()[0], 'datasets': {}}
    for name, spec in DATASETS.items():
        rng = np.random.default_rng(SEED)
        refs = {r['case_id']: r['reference'] for r in read(spec['refs'])}
        ids = sorted(refs)
        judged = {j: {a: {r['case_id']: r for r in read(str(p).format(a=a))} for a in ARMS}
                  for j, p in spec['judges'].items()}
        notes = judged['gpt-4o-mini']
        rouge = {a: {} for a in ARMS}
        for a in ARMS:
            for cid in ids:
                s = scorer.score(refs[cid], flat(notes[a][cid]['output']))
                rouge[a][cid] = {k: s[k].fmeasure for k in s}
        res = {'n': len(ids), 'arms': {}, 'contrasts': {}, 'valid_only': {}}
        for a in ARMS:
            res['arms'][a] = {k: float(np.mean([rouge[a][c][k] for c in ids])) for k in ('rouge1', 'rouge2', 'rougeL')}
            res['arms'][a]['schema_valid'] = sum(int(bool(int(notes[a][c].get('schema_valid') or 0))) for c in ids)
        for metric in ('rougeL', 'rouge1'):
            rows = []
            for a in PRIMARY + ['random_rag', 'schema']:
                d = [rouge[a][c][metric] - rouge['base'][c][metric] for c in ids]
                rows.append(dict(arm=a, diff=float(np.mean(d)), ci95=boot(d, rng), p=float(wilcoxon(d).pvalue)))
            adj = holm([r['p'] for r in rows[:3]])
            for r, q in zip(rows[:3], adj):
                r['p_holm'] = q
            res['contrasts'][metric] = rows
        for j in judged:
            rows = []
            for a in PRIMARY:
                both = [c for c in ids if int(notes['base'][c].get('schema_valid') or 0) and int(notes[a][c].get('schema_valid') or 0)]
                x = [int(judged[j]['base'][c]['faithfulness'] or 0) for c in both]
                y = [int(judged[j][a][c]['faithfulness'] or 0) for c in both]
                up, down, p = mcnemar(x, y)
                rows.append(dict(arm=a, n_both_valid=len(both), base_supported=sum(x), arm_supported=sum(y),
                                 improved=up, worsened=down, p=p))
            for r, q in zip(rows, holm([r['p'] for r in rows])):
                r['p_holm'] = q
            res['valid_only'][j] = rows
        refrouge = {a: res['arms'][a]['rougeL'] for a in ARMS}
        res['note'] = 'ROUGE is against the dataset reference note; source support is from the judges.'
        out['datasets'][name] = res
        print(name, {a: round(v, 3) for a, v in refrouge.items()})
    Path(R / 'reference_overlap_v1.json').write_text(json.dumps(out, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
