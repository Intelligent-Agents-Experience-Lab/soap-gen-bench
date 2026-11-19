"""Rescore the held-out outputs with the frozen judge and run the reference-note control.

Run from the repository root (needs OPENAI_API_KEY; TYPE_SAFE_JEV_API_KEY for --jev-references):

    python -m source.experiments.revision_rescore --out revision/results/rescore_v1
    python -m source.experiments.revision_rescore --out revision/results/rescore_v1 --jev-references

Steps (each is skipped if its output already exists, so the command is resumable):
1. Strip earlier judge fields from the six jev_<condition>_v1.jsonl generation records.
2. Build reference-note inputs: the 62 test references and the 558 training references,
   each scored against its own conversation. Training references are what the LoRA
   adapter imitates and what the retriever returns, so their faithfulness is the
   control for the faithfulness drop seen in the adapted and retrieval arms.
3. Score every input with the frozen judge in revision/protocol/config.json.
4. Optionally score the references with Jev (judge-dependence check).
5. Write paired statistics for the four primary arms and a summary JSON.
Missing judgments stay missing; statistics are refused until they are resolved.
"""
import argparse
import json
from pathlib import Path

from scipy.stats import binomtest

from .revision_core import CONDITIONS, read_jsonl, write_json, write_jsonl
from .revision_runner import judge, load_config
from .revision_statistics import analyze

JUDGE_KEYS = {'faithfulness', 'answer_relevancy', 'faithfulness_probability', 'answer_relevancy_probability',
              'rationale', 'judge_threshold', 'judge_usage'}
REFERENCES = ('reference_test', 'reference_train')


def strip(row):
    return {k: v for k, v in row.items() if k not in JUDGE_KEYS and not k.startswith('judge_')}


def build_inputs(results, inputs, data_dir):
    for condition in CONDITIONS:
        target = inputs / f'{condition}.jsonl'
        if not target.exists():
            rows = [strip(r) for r in read_jsonl(results / f'jev_{condition}_v1.jsonl')]
            if len(rows) != 62 or len({r['case_id'] for r in rows}) != 62:
                raise ValueError(f'Expected 62 unique cases for {condition}')
            write_jsonl(target, rows)
    for condition, split in zip(REFERENCES, ('test', 'train')):
        target = inputs / f'{condition}.jsonl'
        if not target.exists():
            write_jsonl(target, [dict(case_id=r['case_id'], condition=condition, status='ok',
                                      conversation=r['conversation'], output=r['reference'])
                                 for r in read_jsonl(Path(data_dir) / f'{split}.jsonl')])


def kappa(a, b):
    n = len(a)
    if not n:
        return None
    po = sum(x == y for x, y in zip(a, b)) / n
    pa, pb = sum(a) / n, sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return None if pe == 1 else (po - pe) / (1 - pe)


def mcnemar(rows, metric, left, right):
    by = {}
    for r in rows:
        by.setdefault(r['case_id'], {})[r['condition']] = r[metric]
    pairs = [(v[left], v[right]) for v in by.values() if left in v and right in v]
    pos = sum(1 for a, b in pairs if a == 0 and b == 1)
    neg = sum(1 for a, b in pairs if a == 1 and b == 0)
    return dict(contrast=f'{right}_minus_{left}', n=len(pairs),
                difference=sum(b - a for a, b in pairs) / len(pairs), discordant=[pos, neg],
                p=float(binomtest(pos, pos + neg).pvalue) if pos + neg else 1.0, test='exact McNemar')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config', type=Path, default=Path('revision/protocol/config.json'))
    parser.add_argument('--results', type=Path, default=Path('revision/results'))
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--jev-references', action='store_true')
    parser.add_argument('--judge', help='Additional judge model (post hoc); default is the frozen judge in config.json')
    parser.add_argument('--reasoning-effort', choices=['minimal', 'low', 'medium', 'high'],
                        help='Fixed reasoning effort for GPT-5-family judges')
    parser.add_argument('--compare', type=Path, default=Path('revision/results/rescore_v1/scored'),
                        help='Scored folder of the frozen judge, for agreement')
    args = parser.parse_args()
    config, audit = load_config(args.config)
    frozen = config['judge']
    if args.judge and args.judge != frozen:
        config = dict(config, judge=args.judge)
        if args.out.resolve() == args.compare.parent.resolve():
            raise ValueError('Write an additional judge to a new --out folder')
    inputs, scored = args.out / 'inputs', args.out / 'scored'
    build_inputs(args.results, inputs, config['data_dir'])

    for condition in CONDITIONS + REFERENCES:
        target = scored / f'{condition}.jsonl'
        if not target.exists():
            print(f'judging {condition} with {config["judge"]} ...', flush=True)
            judge(config, inputs / f'{condition}.jsonl', target, reasoning_effort=args.reasoning_effort)
    if args.jev_references:
        from .jev_judge import score as jev_score
        for condition in REFERENCES:
            target = scored / f'jev_{condition}.jsonl'
            if not target.exists():
                print(f'judging {condition} with Jev ...', flush=True)
                jev_score(inputs / f'{condition}.jsonl', target, question_set='v1')  # matches jev_*_v1 scoring

    summary = dict(judge_requested=config['judge'], frozen_primary_judge=frozen,
                   role='primary (prespecified)' if config['judge'] == frozen else 'additional judge (post hoc)',
                   reasoning_effort=args.reasoning_effort, conditions=[], references=[], agreement_with_jev=[],
                   agreement_with_frozen_judge=[])
    all_rows = {}
    for condition in CONDITIONS + REFERENCES:
        rows = read_jsonl(scored / f'{condition}.jsonl')
        all_rows[condition] = rows
        ok = [r for r in rows if r.get('faithfulness') is not None]
        entry = dict(condition=condition, n=len(rows), scored=len(ok), missing=len(rows) - len(ok),
                     judge_returned=sorted({r.get('judge_returned') for r in ok if r.get('judge_returned')}),
                     faithfulness=sum(r['faithfulness'] for r in ok),
                     answer_relevancy=sum(r['answer_relevancy'] for r in ok))
        entry['faithfulness_rate'] = entry['faithfulness'] / len(ok) if ok else None
        entry['answer_relevancy_rate'] = entry['answer_relevancy'] / len(ok) if ok else None
        summary['references' if condition in REFERENCES else 'conditions'].append(entry)
    for condition in CONDITIONS:
        jev = {r['case_id']: r for r in read_jsonl(args.results / f'jev_{condition}_v1.jsonl')}
        for metric in ('faithfulness', 'answer_relevancy'):
            pairs = [(r[metric], jev[r['case_id']][metric]) for r in all_rows[condition]
                     if r.get(metric) is not None and jev[r['case_id']].get(metric) is not None]
            summary['agreement_with_jev'].append(dict(
                condition=condition, metric=metric, n=len(pairs),
                percent_agreement=sum(a == b for a, b in pairs) / len(pairs) if pairs else None,
                cohen_kappa=kappa([a for a, _ in pairs], [b for _, b in pairs])))
    if config['judge'] != frozen and args.compare.exists():
        for condition in CONDITIONS + REFERENCES:
            ref = {r['case_id']: r for r in read_jsonl(args.compare / f'{condition}.jsonl')}
            pairs = [(r['faithfulness'], ref[r['case_id']]['faithfulness']) for r in all_rows[condition]
                     if r.get('faithfulness') is not None and ref.get(r['case_id'], {}).get('faithfulness') is not None]
            summary['agreement_with_frozen_judge'].append(dict(
                condition=condition, metric='faithfulness', n=len(pairs),
                percent_agreement=sum(a == b for a, b in pairs) / len(pairs) if pairs else None,
                cohen_kappa=kappa([a for a, _ in pairs], [b for _, b in pairs])))
    if args.jev_references:
        for condition in REFERENCES:
            rows = [r for r in read_jsonl(scored / f'jev_{condition}.jsonl') if r.get('faithfulness') is not None]
            summary['references'].append(dict(condition=f'jev_{condition}', scored=len(rows),
                                              faithfulness=sum(r['faithfulness'] for r in rows),
                                              faithfulness_rate=sum(r['faithfulness'] for r in rows) / len(rows) if rows else None))

    missing = sum(e['missing'] for e in summary['conditions'])
    if missing:
        summary['statistics'] = f'NOT RUN: {missing} missing judgments; rerun after resolving them'
    else:
        matrix = [r for c in CONDITIONS for r in all_rows[c]]
        summary['statistics'] = {}
        for metric in ('faithfulness', 'answer_relevancy'):
            summary['statistics'][metric] = analyze(matrix, metric, audit['test_clusters'])
            summary['statistics'][metric]['secondary_controls'] = [
                mcnemar(matrix, metric, 'rag', 'random_rag'), mcnemar(matrix, metric, 'base', 'schema')]
    target = args.out / 'summary.json'
    if target.exists():
        target = args.out / 'summary_rerun.json'
    write_json(target, summary)
    print(json.dumps({k: summary[k] for k in ('conditions', 'references')}, indent=1))
    print(f'wrote {target}')


if __name__ == '__main__':
    main()
