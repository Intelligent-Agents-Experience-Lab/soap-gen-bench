"""Blinded clinical validation: import the two rater workbooks, check them, and report
inter-rater agreement and agreement of every automated measure with the human labels.

    python -m source.experiments.revision_human_validation \
        --rater student_A=PATH_A.xlsx --rater student_B=PATH_B.xlsx \
        [--adjudicated PATH.xlsx] [--dry-run] --out revision/results/human_validation_v1

--dry-run marks every output as NOT human validation (for testing the pipeline on placeholder or
AI-generated sheets). Without an adjudicated workbook, the reference label is the raters' consensus
(notes where they agree on unsupported claims); disagreements are listed for adjudication.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from .revision_core import read_jsonl
from .revision_measurement_audit_v2 import unsupported

PACKET = Path('revision/results/annotation_packet_v1')
FIELDS = {'unsupported_claims': 3, 'important_omissions': 4, 'misplaced_content': 5, 'severity': 6, 'acceptable': 7,
          'rationale': 8}
BINARY = ('unsupported_claims', 'important_omissions', 'misplaced_content', 'acceptable')
SEVERITY = ('none', 'minor', 'clinically_significant', 'safety_critical')
ARMS = ('base', 'rag', 'lora', 'lora_rag')
JUDGES = {'gpt-4o-mini-2024-07-18 (prespecified)': 'revision/results/rescore_v1/scored/{c}.jsonl',
          'gpt-5.6-luna': 'revision/results/rescore_luna_v1/scored/{c}.jsonl',
          'Jev v1': 'revision/results/jev_{c}_v1.jsonl',
          'Jev v2': 'revision/results/jev_v2/{c}.jsonl'}


def read_sheet(path):
    from openpyxl import load_workbook
    wb = load_workbook(path, data_only=True)
    ws, rater = wb['Ratings'], wb['Rater']
    meta = {str(rater.cell(r, 1).value): rater.cell(r, 2).value for r in range(3, 13) if rater.cell(r, 1).value}
    rows, problems = {}, []
    for r in range(2, ws.max_row + 1):
        nid = ws.cell(r, 1).value
        if not nid:
            continue
        rec = {k: ws.cell(r, c).value for k, c in FIELDS.items()}
        for k in BINARY:
            v = rec[k]
            rec[k] = int(v) if v in (0, 1, '0', '1') else None
            if rec[k] is None:
                problems.append(f'{nid}: {k} missing or invalid ({v!r})')
        if rec['severity'] not in SEVERITY:
            problems.append(f"{nid}: severity missing or invalid ({rec['severity']!r})")
        if not str(rec['rationale'] or '').strip():
            problems.append(f'{nid}: rationale empty')
        rows[nid] = rec
    return rows, meta, problems


def kappa(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    po = float((a == b).mean())
    pa, pb = a.mean(), b.mean()
    pe = pa * pb + (1 - pa) * (1 - pb)
    return po, (None if pe == 1 else float((po - pe) / (1 - pe)))


def boot(a, b, clusters, fn, samples=10000, seed=3407):
    groups = {}
    for i, c in enumerate(clusters):
        groups.setdefault(c, []).append(i)
    keys = list(groups); rng = np.random.default_rng(seed); vals = []
    a, b = np.asarray(a), np.asarray(b)
    for _ in range(samples):
        idx = np.concatenate([groups[keys[j]] for j in rng.integers(len(keys), size=len(keys))])
        v = fn(a[idx], b[idx])
        if v is not None:
            vals.append(v)
    return [float(x) for x in np.quantile(vals, [.025, .975])] if vals else None


def agreement(a, b, clusters):
    po, k = kappa(a, b)
    return dict(n=len(a), raw_agreement=round(po, 3), kappa=None if k is None else round(k, 3),
                kappa_ci95=boot(a, b, clusters, lambda x, y: kappa(x, y)[1]),
                positives=[int(sum(a)), int(sum(b))])


def detect(human_unsupported, judge_supported):
    """Judge as a detector of unsupported content: positive = human says unsupported."""
    h = np.asarray(human_unsupported); j = 1 - np.asarray(judge_supported)
    tp = int(((h == 1) & (j == 1)).sum()); fn = int(((h == 1) & (j == 0)).sum())
    tn = int(((h == 0) & (j == 0)).sum()); fp = int(((h == 0) & (j == 1)).sum())
    return dict(tp=tp, fn=fn, tn=tn, fp=fp, sensitivity=round(tp / (tp + fn), 3) if tp + fn else None,
                specificity=round(tn / (tn + fp), 3) if tn + fp else None)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--rater', action='append', required=True, metavar='NAME=PATH',
                   help='exactly two independent raters, e.g. --rater student_A=path.xlsx')
    p.add_argument('--adjudicated', type=Path)
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--out', type=Path, default=Path('revision/results/human_validation_v1'))
    a = p.parse_args()
    key = {k['note_id']: k for k in json.loads((PACKET / 'annotation_key.json').read_text(encoding='utf-8'))}
    enc = {k['note_id']: k['encounter'] for k in json.loads((PACKET / 'encounter_map.json').read_text(encoding='utf-8'))}
    packet = {r['note_id']: r for r in json.loads((PACKET / 'ratings.json').read_text(encoding='utf-8'))}
    if len(a.rater) != 2:
        raise SystemExit('Give exactly two --rater NAME=PATH')
    (na, pa_), (nb, pb_) = [r.split('=', 1) for r in a.rater]
    cl, cl_meta, cl_prob = read_sheet(Path(pa_))
    st, st_meta, st_prob = read_sheet(Path(pb_))
    ids = sorted(key)
    if set(cl) != set(ids) or set(st) != set(ids):
        raise SystemExit('Workbooks do not cover exactly the 80 packet notes')
    # packet notes must be the generated outputs that the judges scored
    outputs = {}
    for arm in ARMS:
        for r in read_jsonl(Path(f'revision/results/rescore_v1/inputs/{arm}.jsonl')):
            outputs[(r['case_id'], arm)] = r['output']
    mismatched = [n for n in ids if outputs[(key[n]['case_id'], key[n]['condition'])].strip() != packet[n]['note'].strip()]
    if a.adjudicated:
        ref_src, ref = 'adjudicated workbook', read_sheet(a.adjudicated)[0]
        ref_ids = ids
    else:
        ref_src = f'consensus of {na} and {nb}: notes where both agree on unsupported_claims (disagreements await adjudication)'
        ref = cl
        ref_ids = [n for n in ids if cl[n]['unsupported_claims'] is not None and cl[n]['unsupported_claims'] == st[n]['unsupported_claims']]
    clusters = [enc[n] for n in ids]
    out = dict(status='DRY RUN - NOT HUMAN VALIDATION; do not report' if a.dry_run else 'human ratings',
               reference_label=ref_src, raters={na: cl_meta, nb: st_meta},
               data_problems={na: cl_prob, nb: st_prob}, packet_notes_not_matching_scored_outputs=mismatched,
               inter_rater={}, by_arm={}, automated_vs_reference={})
    complete = [n for n in ids if all(cl[n][k] is not None and st[n][k] is not None for k in BINARY)]
    cc = [enc[n] for n in complete]
    for k in BINARY:
        out['inter_rater'][k] = agreement([cl[n][k] for n in complete], [st[n][k] for n in complete], cc)
    sev = [(cl[n]['severity'], st[n]['severity']) for n in complete]
    out['inter_rater']['severity_exact_agreement'] = round(sum(x == y for x, y in sev) / len(sev), 3) if sev else None
    for arm in ARMS:
        notes = [n for n in ids if key[n]['condition'] == arm]
        out['by_arm'][arm] = {who: {**{k: sum(int(r[n][k] or 0) for n in notes) for k in BINARY},
                                    'safety_critical': sum(r[n]['severity'] == 'safety_critical' for n in notes),
                                    'clinically_significant': sum(r[n]['severity'] == 'clinically_significant' for n in notes),
                                    'n': len(notes)}
                              for who, r in ((na, cl), (nb, st))}
    # human "supported" = substantive note and no unsupported claim (rubric construct)
    human_unsup = {n: 1 if ref[n]['unsupported_claims'] == 1 or not packet[n]['note'].strip() else 0 for n in ref_ids}
    out['reference_n'] = len(ref_ids)
    out['disagreements_unsupported'] = [n for n in ids if n not in ref_ids]
    for name, pat in JUDGES.items():
        by = {}
        for arm in ARMS:
            for r in read_jsonl(Path(pat.format(c=arm))):
                by[(r['case_id'], arm)] = r.get('faithfulness')
        jn = [n for n in ref_ids if by.get((key[n]['case_id'], key[n]['condition'])) is not None]
        js = [by[(key[n]['case_id'], key[n]['condition'])] for n in jn]
        hs = [1 - human_unsup[n] for n in jn]
        out['automated_vs_reference'][name] = dict(agreement(hs, js, [enc[n] for n in jn]),
                                                   **detect([human_unsup[n] for n in jn], js))
    au = [0 if unsupported(packet[n]['note'], packet[n]['conversation']) else 1 for n in ref_ids]
    out['automated_vs_reference']['numeric audit v2 (judge-free)'] = dict(
        agreement([1 - human_unsup[n] for n in ref_ids], au, [enc[n] for n in ref_ids]), **detect([human_unsup[n] for n in ref_ids], au),
        note='The audit detects only numeric content, so low sensitivity is expected.')
    a.out.mkdir(parents=True, exist_ok=True)
    name = 'DRY_RUN_report.json' if a.dry_run else 'report.json'
    (a.out / name).write_text(json.dumps(out, indent=2, default=str), encoding='utf-8')
    print(json.dumps({k: out[k] for k in ('status', 'reference_label', 'reference_n', 'packet_notes_not_matching_scored_outputs')}, indent=1))
    print('raters:', json.dumps(out['raters'], default=str)[:600])
    print('problems:', len(cl_prob), len(st_prob))
    for k, v in out['inter_rater'].items():
        print('inter-rater', k, v)
    for arm, v in out['by_arm'].items():
        print(arm, v)
    for k, v in out['automated_vs_reference'].items():
        print(k, {x: v[x] for x in ('raw_agreement', 'kappa', 'kappa_ci95', 'sensitivity', 'specificity', 'tp', 'fn', 'tn', 'fp')})


if __name__ == '__main__':
    main()
