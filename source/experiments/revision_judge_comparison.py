"""Combine all judges and the measurement audit into one comparison file.

    python -m source.experiments.revision_judge_comparison --output revision/results/judge_comparison_v1.json
"""
import argparse
import json
from pathlib import Path

from scipy.stats import binomtest

from .revision_core import CONDITIONS, read_jsonl, write_json
from .revision_rescore import kappa
from .revision_statistics import analyze

R = Path('revision/results')
SOURCES = {
    'gpt-4o-mini (primary)': R / 'rescore_v1/scored/{g}.jsonl',
    'gpt-5.6-luna': R / 'rescore_luna_v1/scored/{g}.jsonl',
    'jev v1': None,  # condition files jev_<g>_v1.jsonl; references in rescore_v1/scored/jev_reference_*.jsonl
    'jev v2': R / 'jev_v2/{g}.jsonl',
}
GROUPS = CONDITIONS + ('reference_test', 'reference_train')


def path(judge, group):
    if judge == 'jev v1':
        return R / (f'jev_{group}_v1.jsonl' if group in CONDITIONS else f'rescore_v1/scored/jev_{group}.jsonl')
    return Path(str(SOURCES[judge]).format(g=group))


def load(judge, group):
    return {r['case_id']: r['faithfulness'] for r in read_jsonl(path(judge, group))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--audit', type=Path, default=R / 'data_v1/audit.json')
    args = parser.parse_args()
    clusters = json.loads(args.audit.read_text(encoding='utf-8'))['test_clusters']
    data = {j: {g: load(j, g) for g in GROUPS} for j in SOURCES}
    out = dict(counts={}, primary_contrasts={}, secondary_vs_base={}, agreement=[])
    for j in SOURCES:
        out['counts'][j] = {g: sum(data[j][g].values()) for g in GROUPS}
        rows = [dict(case_id=k, condition=c, faithfulness=v) for c in CONDITIONS for k, v in data[j][c].items()]
        out['primary_contrasts'][j] = analyze(rows, 'faithfulness', clusters)['contrasts']
        sec = []
        for arm in ('random_rag', 'schema'):
            pos = sum(1 for k, v in data[j][arm].items() if v == 1 and data[j]['base'][k] == 0)
            neg = sum(1 for k, v in data[j][arm].items() if v == 0 and data[j]['base'][k] == 1)
            sec.append(dict(contrast=f'{arm}_minus_base', discordant=[pos, neg],
                            p=float(binomtest(pos, pos + neg).pvalue) if pos + neg else 1.0))
        rr = [(data[j]['rag'][k], data[j]['random_rag'][k]) for k in data[j]['rag']]
        pos, neg = sum(1 for a, b in rr if a == 0 and b == 1), sum(1 for a, b in rr if a == 1 and b == 0)
        sec.append(dict(contrast='random_rag_minus_rag', discordant=[pos, neg],
                        p=float(binomtest(pos, pos + neg).pvalue) if pos + neg else 1.0))
        out['secondary_vs_base'][j] = sec
    judges = list(SOURCES)
    for i, a in enumerate(judges):
        for b in judges[i + 1:]:
            for g in GROUPS:
                pairs = [(data[a][g][k], data[b][g][k]) for k in data[a][g] if k in data[b][g]]
                out['agreement'].append(dict(judges=[a, b], group=g, n=len(pairs),
                                             percent_agreement=sum(x == y for x, y in pairs) / len(pairs),
                                             cohen_kappa=kappa([x for x, _ in pairs], [y for _, y in pairs])))
    v2 = {g: read_jsonl(path('jev v2', g)) for g in GROUPS}
    out['jev_v2_defects_likely'] = {g: {q: sum(r['defects'][q] == 'yes' for r in rows if r.get('defects'))
                                        for q in rows[0]['defects']} for g, rows in v2.items()}
    out['jev_v2_bands'] = {g: {b: sum(r.get('faithfulness_band') == b for r in rows)
                               for b in ('supported', 'indeterminate', 'unsupported')} for g, rows in v2.items()}
    out['measurement_audit'] = json.loads((R / 'measurement_audit_v1.json').read_text(encoding='utf-8'))['groups']
    for g in out['measurement_audit'].values():
        g.pop('per_case', None)
    write_json(args.output, out)
    print(json.dumps(out['counts'], indent=1))
    for j, cs in out['primary_contrasts'].items():
        print(j, [(c['contrast'], round(c['difference'], 3), [round(x, 2) for x in c['ci95']], f"{c['p_holm']:.2g}") for c in cs])
    for j, cs in out['secondary_vs_base'].items():
        print(j, [(c['contrast'], c['discordant'], f"{c['p']:.2g}") for c in cs])
    for a in out['agreement']:
        if a['group'] in ('base', 'rag', 'lora', 'reference_test'):
            print(a['judges'], a['group'], round(a['percent_agreement'], 2), None if a['cohen_kappa'] is None else round(a['cohen_kappa'], 2))


if __name__ == '__main__':
    main()
