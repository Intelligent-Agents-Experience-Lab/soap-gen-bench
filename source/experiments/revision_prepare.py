"""Download pinned public splits and produce the revision contamination audit."""
import argparse
from pathlib import Path
from .revision_core import audit_splits, digest, record, soap_json, write_json, write_jsonl

DATASET = 'adesouza1/soap_notes'
REVISION = '11084a66043aca1cf8809a839d5d43881c229afb'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('revision/results/data_v1'))
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit('Output exists; use a new version directory to preserve provenance')
    from datasets import load_dataset
    dataset = load_dataset(DATASET, revision=REVISION, token=False)
    train = [record(r, 'train', i) for i, r in enumerate(dataset['train'])]
    test = [record(r, 'test', i) for i, r in enumerate(dataset['test'])]
    audit, retained = audit_splits(train, test)
    sft, conversion_failures = [], []
    for row in retained:
        try:
            sft.append(dict(row, target=soap_json(row['reference'])))
        except ValueError as exc:
            conversion_failures.append(dict(case_id=row['case_id'], reason=str(exc)))
    write_jsonl(args.output / 'train.jsonl', retained)
    write_jsonl(args.output / 'test.jsonl', test)
    write_jsonl(args.output / 'sft.jsonl', sft)
    audit.update(dataset=DATASET, dataset_revision=REVISION, sft_count=len(sft),
                 conversion_failures=conversion_failures,
                 status='AUDIT_ONLY_NOT_FROZEN', near_duplicate_review='pending',
                 fingerprints={name: digest((args.output / name).read_text(encoding='utf-8'))
                               for name in ('train.jsonl', 'test.jsonl', 'sft.jsonl')})
    write_json(args.output / 'audit.json', audit)
    print({k: audit[k] for k in ('train_count', 'test_count', 'retained_train_count', 'sft_count', 'status')})
    print('Flagged cross-split pairs:', len(audit['flagged_pairs']))
    print('Invalid rows:', len(audit['invalid_ids']))


if __name__ == '__main__':
    main()
