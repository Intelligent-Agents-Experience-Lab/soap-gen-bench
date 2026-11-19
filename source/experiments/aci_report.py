"""ACI-BENCH replication report: every arm, judge and audit, paired statistics, the locked decision rule,
and verbatim copying from retrieved notes.  Writes revision/results/aci/report_v1.json (no case text).

    python -m source.experiments.aci_report
"""
import json
from pathlib import Path

from scipy.stats import binomtest

from .revision_core import CONDITIONS, read_jsonl, shingles
from .revision_measurement_audit_v2 import unsupported
from .revision_statistics import analyze, holm

R = Path('revision/results/aci')
DATA = Path('revision/results/data_aci_v1')
JUDGES = {'gpt4omini': 'gpt-4o-mini-2024-07-18 (prespecified)', 'luna': 'gpt-5.6-luna (post hoc)',
          'jevv2': 'Jev v2 (post hoc)'}
KEY_DEFECTS = ('unsupported_measurement', 'unsupported_exam_finding', 'unsupported_plan')


def mcnemar(a, b):
    pos = sum(1 for k in a if a[k] == 0 and b[k] == 1)
    neg = sum(1 for k in a if a[k] == 1 and b[k] == 0)
    return dict(discordant=[pos, neg], p=float(binomtest(pos, pos + neg).pvalue) if pos + neg else 1.0)


def main():
    audit = json.loads((DATA / 'audit.json').read_text(encoding='utf-8'))
    clusters = audit['test_clusters']
    groups = ('reference',) + CONDITIONS
    gens = {g: read_jsonl(R / ('reference_inputs.jsonl' if g == 'reference' else f'generations_{g}.jsonl')) for g in groups}
    n = len(gens['base'])
    out = dict(dataset='ACI-BENCH test1-3', n_test=n, n_train=audit['retained_train_count'], arms={}, contrasts={},
               controls={}, decision_rule={}, copying={})
    flags = {}
    for g in groups:
        rows = gens[g]
        flags[('audit', g)] = {r['case_id']: int(not unsupported(r['output'], r['conversation'])) for r in rows}
        arm = dict(n=len(rows), numeric_audit_flagged=sum(1 - v for v in flags[('audit', g)].values()))
        if g != 'reference':
            arm.update(schema_valid=sum(int(bool(r.get('schema_valid'))) for r in rows),
                       hit_output_limit=sum(int(bool(r.get('hit_output_limit'))) for r in rows),
                       failures=sum(r.get('status') != 'ok' for r in rows),
                       mean_output_tokens=round(sum(r.get('output_tokens', 0) for r in rows) / len(rows), 1),
                       torch=rows[0]['packages'].get('torch'), transformers=rows[0]['packages'].get('transformers'))
        for j in JUDGES:
            jr = read_jsonl(R / f'{j}_{g}.jsonl')
            flags[(j, g)] = {r['case_id']: r['faithfulness'] for r in jr}
            arm[f'{j}_supported'] = sum(v or 0 for v in flags[(j, g)].values())
            arm[f'{j}_missing'] = sum(v is None for v in flags[(j, g)].values())
            if j == 'jevv2':
                arm['jevv2_defects_likely'] = {q: sum(r['defects'][q] == 'yes' for r in jr if r.get('defects'))
                                               for q in jr[0]['defects']}
        out['arms'][g] = arm
    for m in list(JUDGES) + ['audit']:
        rows = [dict(case_id=k, condition=c, value=v) for c in ('base', 'rag', 'lora', 'lora_rag')
                for k, v in flags[(m, c)].items()]
        res = analyze(rows, 'value', clusters, adjust_interaction=False)
        out['contrasts'][m] = res
        out['controls'][m] = {'random_rag_vs_rag': mcnemar(flags[(m, 'rag')], flags[(m, 'random_rag')]),
                              'random_rag_vs_base': mcnemar(flags[(m, 'base')], flags[(m, 'random_rag')]),
                              'schema_vs_base': mcnemar(flags[(m, 'base')], flags[(m, 'schema')])}
    ref, lora = out['arms']['reference'], out['arms']['lora']
    pct = lambda arm, d: 100 * arm['jevv2_defects_likely'][d] / arm['n']
    diffs = {d: round(pct(lora, d) - pct(ref, d), 1) for d in KEY_DEFECTS}
    r_ref = 100 * ref['numeric_audit_flagged'] / n
    r_lora = 100 * lora['numeric_audit_flagged'] / n
    supports = r_lora <= r_ref + 10 and all(abs(v) <= 15 for v in diffs.values())
    contradicts = r_lora > r_ref + 10 or sum(v > 15 for v in diffs.values()) >= 2
    out['decision_rule'] = dict(plan='revision/protocol/aci_analysis_plan.md (locked 2026-10-31)',
                                reference_numeric_pct=round(r_ref, 1), lora_numeric_pct=round(r_lora, 1),
                                jevv2_lora_minus_reference_pct=diffs,
                                verdict='supports' if supports else 'contradicts' if contradicts else 'inconclusive')
    contexts = {r['case_id']: r for r in read_jsonl(R / 'contexts.jsonl')}
    for g in ('base', 'rag', 'random_rag', 'lora', 'lora_rag', 'schema'):
        key = 'random_rag' if g == 'random_rag' else 'rag'
        share_ctx, share_conv = [], []
        for r in gens[g]:
            o = shingles(r['output'])
            if not o:
                continue
            share_ctx.append(len(o & shingles(contexts[r['case_id']][key]['context'])) / len(o))
            share_conv.append(len(o & shingles(r['conversation'])) / len(o))
        share_ctx.sort(); share_conv.sort()
        med = lambda xs: round(xs[len(xs) // 2], 3)
        out['copying'][g] = dict(median_share_of_note_5grams_in_retrieved_notes=med(share_ctx),
                                 median_share_in_conversation=med(share_conv),
                                 notes_with_over_25pct_copied=sum(x > .25 for x in share_ctx))
    out['copying']['note'] = ('Retrieved notes = the cached context actually shown (similar retrieval); random_rag uses its own '
                              'random context. Base, LoRA and schema never see a context; their values measure chance overlap '
                              'with the same notes.')
    main_ctx = {r['case_id']: r for r in read_jsonl(Path('revision/results/contexts_v5.jsonl'))}
    out['copying_main_dataset'] = {}
    for g in ('base', 'rag', 'random_rag', 'lora', 'lora_rag', 'schema'):
        key = 'random_rag' if g == 'random_rag' else 'rag'
        xs = sorted(len(shingles(r['output']) & shingles(main_ctx[r['case_id']][key]['context'])) / len(shingles(r['output']))
                    for r in read_jsonl(Path(f'revision/results/rescore_v1/inputs/{g}.jsonl')) if shingles(r['output']))
        out['copying_main_dataset'][g] = dict(median_share_of_note_5grams_in_retrieved_notes=round(xs[len(xs) // 2], 3),
                                              notes_with_over_25pct_copied=sum(x > .25 for x in xs), n=len(xs))
    (R / 'report_v1.json').write_text(json.dumps(out, indent=2), encoding='utf-8')
    print(json.dumps({g: {k: v for k, v in a.items() if k != 'jevv2_defects_likely'} for g, a in out['arms'].items()}, indent=0))
    print(out['decision_rule'])
    print({g: c for g, c in out['copying'].items() if g != 'note'})
    for m, res in out['contrasts'].items():
        print(m, [(c['contrast'], round(c['difference'], 3), [round(x, 3) for x in c['ci95']],
                   None if c['p_holm'] is None else round(c['p_holm'], 4), c['test']) for c in res['contrasts']])
    print(json.dumps(out['controls'], indent=0)[:1500])


if __name__ == '__main__':
    main()
