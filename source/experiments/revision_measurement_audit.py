"""Post hoc, judge-free audit: measurements in a note that never appear in its conversation.

A "measurement" is a blood-pressure-style ratio (e.g. 120/80) or a number followed by a
clinical unit (mg, mg/dL, mmHg, bpm, %, degrees F/C, kg, lbs, ...). A measurement is
unsupported when its numeric part does not occur anywhere in the conversation text.
This is a deliberately narrow, reproducible check: it catches invented vitals, labs and
doses, not unsupported narrative findings, and it can miss values written as words.

    python -m source.experiments.revision_measurement_audit --inputs revision/results/rescore_v1/inputs --output revision/results/measurement_audit_v1.json
"""
import argparse
import json
import re
from pathlib import Path

from scipy.stats import binomtest

from .revision_core import CONDITIONS, read_jsonl, write_json

UNIT = (r'(?:mg/dl|mg|mcg|g/dl|mmhg|bpm|beats per minute|breaths per minute|%|°f|°c|degrees|'
        r'kg|lbs?|pounds|cm|mm|ml|units|mmol/l|ng/ml|iu)')
PATTERN = re.compile(r'\b\d{2,3}\s*/\s*\d{2,3}\b|\b\d+(?:\.\d+)?\s*' + UNIT + r'(?![a-z])', re.I)
GROUPS = CONDITIONS + ('reference_test', 'reference_train')


def note_text(output):
    try:
        data = json.loads(output)
    except (TypeError, ValueError):
        return output
    return ' '.join(str(v) for v in data.values()) if isinstance(data, dict) else output


def unsupported(output, conversation):
    found = {re.sub(r'\s+', '', m.group(0).lower()) for m in PATTERN.finditer(note_text(output))}
    conversation = conversation.lower()
    return sorted(x for x in found if re.match(r'[\d./]+', x).group(0).rstrip('.') not in conversation)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    flags, summary = {}, {}
    for group in GROUPS:
        rows = read_jsonl(args.inputs / f'{group}.jsonl')
        per_case = {r['case_id']: unsupported(r['output'], r['conversation']) for r in rows}
        flags[group] = {k: int(bool(v)) for k, v in per_case.items()}
        summary[group] = dict(n=len(rows), notes_with_unsupported_measurement=sum(flags[group].values()),
                              unsupported_measurements=sum(len(v) for v in per_case.values()),
                              per_case={k: v for k, v in per_case.items() if v} if group in CONDITIONS else None)
    contrasts = []
    for arm in CONDITIONS[1:]:
        pos = sum(1 for k, v in flags[arm].items() if v == 1 and flags['base'][k] == 0)
        neg = sum(1 for k, v in flags[arm].items() if v == 0 and flags['base'][k] == 1)
        contrasts.append(dict(contrast=f'{arm}_minus_base', discordant=[pos, neg],
                              p=float(binomtest(pos, pos + neg).pvalue) if pos + neg else 1.0,
                              test='exact McNemar (post hoc, unadjusted)'))
    write_json(args.output, dict(status='POST_HOC', definition=__doc__.strip().splitlines()[0],
                                 groups=summary, contrasts=contrasts))
    print(json.dumps({g: summary[g]['notes_with_unsupported_measurement'] for g in GROUPS}), json.dumps(contrasts))


if __name__ == '__main__':
    main()
