"""Prepare ACI-BENCH (Yim et al., 2023; CC BY 4.0) for the held-out replication.

Training pool = train + valid (87 encounters); held-out test = the three official test sets
(test1, test2, test3; 120 encounters). Test notes are never indexed or trained on.

ACI-BENCH notes use clinical section headers (CHIEF COMPLAINT, HISTORY OF PRESENT ILLNESS,
PHYSICAL EXAM, RESULTS, ASSESSMENT AND PLAN, ...). They are mapped deterministically, without
rewriting any sentence, into the four SOAP fields used by the main experiment:

  subjective  chief complaint, HPI, review of systems, past/social/family history, medications, allergies
  objective   physical exam, vitals, results, EKG
  assessment  ASSESSMENT / IMPRESSION; in problem-oriented A&P blocks, the problem headings and
              their "Medical Reasoning" bullets
  plan        PLAN / INSTRUCTIONS / ORDERS / PROCEDURE; in problem-oriented A&P blocks, every other
              bullet (testing, treatment, counseling, referrals), prefixed with its problem name

Notes whose A&P cannot be split this way, or that would leave a SOAP field empty, are recorded as
conversion failures: they stay in the retrieval corpus (train) or the test set (with their
original note as reference) but are excluded from fine-tuning targets.

    python -m source.experiments.aci_prepare            # -> revision/results/data_aci_v1
"""
import argparse
import csv
import re
from pathlib import Path

from .revision_core import FIELDS, audit_splits, digest, normalize, write_json, write_jsonl

SOURCE = Path('data/external/aci-bench')
ACI_COMMIT = 'b909b2bb9cf1d19de08df15cddde7bd0179665e4'  # see data/external/aci-bench/SOURCE.md
TRAIN_FILES = ('train.csv', 'valid.csv')
TEST_FILES = ('clinicalnlp_taskB_test1.csv', 'clinicalnlp_taskC_test2.csv', 'clef_taskC_test3.csv')
SUBJECTIVE = {'CHIEF COMPLAINT', 'HISTORY OF PRESENT ILLNESS', 'REVIEW OF SYSTEMS', 'REVIEW OF SYMPTOMS',
              'SOCIAL HISTORY', 'MEDICAL HISTORY', 'PAST MEDICAL HISTORY', 'PAST HISTORY', 'SURGICAL HISTORY',
              'FAMILY HISTORY', 'MEDICATIONS', 'CURRENT MEDICATIONS', 'ALLERGIES', 'BIRTH HISTORY', 'SUBJECTIVE'}
OBJECTIVE = {'PHYSICAL EXAM', 'PHYSICAL EXAMINATION', 'EXAM', 'VITALS', 'VITALS REVIEWED', 'RESULTS', 'EKG'}
ASSESSMENT = {'ASSESSMENT', 'IMPRESSION'}
PLAN = {'PLAN', 'INSTRUCTIONS', 'ORDERS', 'PROCEDURE'}
COMBINED = {'ASSESSMENT AND PLAN'}
HEADERS = SUBJECTIVE | OBJECTIVE | ASSESSMENT | PLAN | COMBINED
LABELS = ('Medical Reasoning', 'Additional Testing', 'Medical Treatment', 'Patient Education and Counseling',
          'Specialist Referrals', 'Patient Agreements')
BULLET = re.compile(r'^\s*[•\-\*]\s*([A-Za-z][A-Za-z /]+?):\s*(.*)$')
INLINE = re.compile(r'\s+[•\-]\s+(?=(?:' + '|'.join(LABELS) + r'):)')


def sections(note):
    """Split a note at known header lines. Text before the first header is returned with header None."""
    out, current, buffer = [], None, []
    for line in note.splitlines():
        if line.strip() in HEADERS:
            out.append((current, '\n'.join(buffer).strip()))
            current, buffer = line.strip(), []
        else:
            buffer.append(line)
    out.append((current, '\n'.join(buffer).strip()))
    return [(h, t) for h, t in out if t]


def split_problem_oriented(text):
    """Split a problem-oriented A&P block: reasoning -> assessment, everything else -> plan."""
    lines = [l.strip() for l in INLINE.sub('\n- ', text).splitlines() if l.strip()]
    assessment, plan, problem = [], [], None
    for i, line in enumerate(lines):
        match = BULLET.match(line)
        if match:
            label, body = match.group(1).strip(), match.group(2).strip()
            if label.lower() in ('medical reasoning', 'assessment', 'impression'):
                assessment.append(f'{label}: {body}')
            else:
                plan.append(f'{problem} - {label}: {body}' if problem else f'{label}: {body}')
        elif line.lower().startswith(('patient agreements', 'follow up', 'follow-up')):
            plan.append(line)
        elif i + 1 < len(lines) and BULLET.match(lines[i + 1]):
            problem = line.rstrip('.:')
            assessment.append(line)
        else:
            assessment.append(line)
    return '\n'.join(assessment), '\n'.join(plan)


def convert(note):
    """Return ({subjective, objective, assessment, plan}, method) or raise ValueError."""
    parts = sections(note)
    fields = {k: [] for k in FIELDS}
    for header, text in parts:
        block = f'{header}\n{text}' if header else text
        if header is None or header in SUBJECTIVE:
            fields['subjective'].append(block)
        elif header in OBJECTIVE:
            fields['objective'].append(block)
        elif header in ASSESSMENT:
            fields['assessment'].append(block)
        elif header in PLAN:
            fields['plan'].append(block)
    ap_text = '\n'.join(t for h, t in parts if h in COMBINED | ASSESSMENT | {'PLAN'})
    if 'Medical Reasoning:' in ap_text:
        assessment, plan = split_problem_oriented(ap_text)
        fields['assessment'] = [assessment]
        fields['plan'] = [plan] + [f'{h}\n{t}' for h, t in parts if h in PLAN - {'PLAN'}]
        method = 'problem_oriented'
    elif any(h in COMBINED for h, _ in parts):
        raise ValueError('Combined ASSESSMENT AND PLAN without problem-oriented bullets')
    else:
        method = 'sections'
    result = {k: '\n\n'.join(x for x in v if x).strip() for k, v in fields.items()}
    empty = [k for k, v in result.items() if not v]
    if empty:
        raise ValueError(f'Empty SOAP field(s): {", ".join(empty)}')
    return result, method


def soap_text(target):
    return '\n\n'.join(f'{k.capitalize()}:\n{target[k]}' for k in FIELDS)


def load(source, files, split):
    csv.field_size_limit(10 ** 9)
    rows = []
    for name in files:
        with open(source / name, encoding='utf-8', newline='') as handle:
            for r in csv.DictReader(handle):
                rows.append(dict(case_id=f'{split}:{r["encounter_id"]}', source_file=name,
                                 subset=r['dataset'], encounter_id=r['encounter_id'],
                                 conversation=r['dialogue'], reference_original=r['note']))
    return rows


def finalize(row):
    try:
        target, method = convert(row['reference_original'])
        row.update(reference=soap_text(target), target=target, conversion=method)
    except ValueError as exc:
        row.update(reference=row['reference_original'], target=None, conversion=f'failed: {exc}')
    row.update(conversation_hash=digest(normalize(row['conversation'])),
               reference_hash=digest(normalize(row['reference'])))
    return row


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--source', type=Path, default=SOURCE)
    parser.add_argument('--output', type=Path, default=Path('revision/results/data_aci_v1'))
    args = parser.parse_args(argv)
    if args.output.exists():
        raise SystemExit('Output exists; use a new version directory to preserve provenance')
    train = [finalize(r) for r in load(args.source, TRAIN_FILES, 'train')]
    test = [finalize(r) for r in load(args.source, TEST_FILES, 'test')]
    ids = [r['encounter_id'] for r in train + test]
    if len(ids) != len(set(ids)):
        raise SystemExit('Duplicate encounter ids across ACI-BENCH files')
    audit, retained = audit_splits(train, test)
    sft = [dict(r) for r in retained if r['target']]
    failures = [dict(case_id=r['case_id'], reason=r['conversion']) for r in train + test if r['target'] is None]
    write_jsonl(args.output / 'train.jsonl', retained)
    write_jsonl(args.output / 'test.jsonl', test)
    write_jsonl(args.output / 'sft.jsonl', sft)
    audit.update(dataset='ACI-BENCH (wyim/aci-bench)', dataset_revision=ACI_COMMIT,
                 source_files=dict(train=list(TRAIN_FILES), test=list(TEST_FILES)),
                 source_fingerprints={n: digest((args.source / n).read_text(encoding='utf-8')) for n in TRAIN_FILES + TEST_FILES},
                 sft_count=len(sft), conversion_failures=failures,
                 conversion_methods={m: sum(r['conversion'] == m for r in train + test) for m in ('problem_oriented', 'sections')},
                 subsets={s: dict(train=sum(r['subset'] == s for r in retained), test=sum(r['subset'] == s for r in test))
                          for s in sorted({r['subset'] for r in train + test})},
                 status='AUDIT_ONLY_NOT_FROZEN',
                 fingerprints={name: digest((args.output / name).read_text(encoding='utf-8'))
                               for name in ('train.jsonl', 'test.jsonl', 'sft.jsonl')})
    write_json(args.output / 'audit.json', audit)
    print({k: audit[k] for k in ('train_count', 'test_count', 'retained_train_count', 'sft_count', 'status')})
    print('Flagged cross-split pairs:', len(audit['flagged_pairs']), '| conversion failures:', len(failures))


if __name__ == '__main__':
    main()
