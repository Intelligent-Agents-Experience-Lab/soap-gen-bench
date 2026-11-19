"""Auditable, dependency-light primitives for the held-out revision."""
import hashlib
import json
import math
import random
import re
from pathlib import Path

FIELDS = ('subjective', 'objective', 'assessment', 'plan')
CONDITIONS = ('base', 'rag', 'lora', 'lora_rag', 'random_rag', 'schema')


def digest(value):
    return hashlib.sha256(value.encode('utf-8')).hexdigest()


def normalize(text):
    return ' '.join(text.casefold().split())


def shingles(text):
    words = normalize(text).split()
    return {tuple(words[i:i + 5]) for i in range(max(1, len(words) - 4))} if words else set()


def similarity(left, right):
    return len(left & right) / len(left | right) if left and right else 0.0


def record(row, split, index):
    conversation = next((row[k] for k in ('patient_convo', 'dialogue', 'conversation', 'text') if row.get(k)), '')
    reference = next((row[k] for k in ('soap_notes', 'soap_note') if row.get(k)), '')
    return dict(case_id=f'{split}:{index}', conversation=conversation, reference=reference,
                conversation_hash=digest(normalize(conversation)), reference_hash=digest(normalize(reference)))


def audit_splits(train, test, threshold=0.80):
    """Conservatively quarantine all flagged training rows before human review."""
    flagged, excluded = [], set()
    all_rows = train + test
    invalid = [r['case_id'] for r in all_rows if not r['conversation'].strip() or not r['reference'].strip()]
    tokens = {r['case_id']: (shingles(r['conversation']), shingles(r['reference'])) for r in all_rows}
    for a in train:
        for b in test:
            scores = [similarity(x, y) for x, y in zip(tokens[a['case_id']], tokens[b['case_id']])]
            exact = any(a[k] == b[k] for k in ('conversation_hash', 'reference_hash'))
            if exact or max(scores) >= threshold:
                excluded.add(a['case_id'])
                flagged.append(dict(train_id=a['case_id'], test_id=b['case_id'], exact=exact,
                                    conversation_similarity=scores[0], reference_similarity=scores[1]))
    # Connected components preserve dependence among duplicate test encounters.
    parent = {r['case_id']: r['case_id'] for r in test}
    def root(key):
        while parent[key] != key:
            key = parent[key]
        return key
    test_pairs = []
    for i, a in enumerate(test):
        for b in test[i + 1:]:
            score = max(similarity(x, y) for x, y in zip(tokens[a['case_id']], tokens[b['case_id']]))
            if score >= threshold:
                parent[root(b['case_id'])] = root(a['case_id'])
                test_pairs.append(dict(left=a['case_id'], right=b['case_id'], similarity=score))
    seen, unique_train, within_train = set(), [], []
    for row in train:
        if row['case_id'] in excluded or row['case_id'] in invalid:
            continue
        if row['conversation_hash'] in seen:
            within_train.append(row['case_id'])
            continue
        seen.add(row['conversation_hash'])
        unique_train.append(row)
    return dict(train_count=len(train), test_count=len(test), retained_train_count=len(unique_train),
                invalid_ids=invalid, excluded_train_ids=sorted(excluded), flagged_pairs=flagged,
                within_train_duplicate_ids=within_train, test_duplicate_pairs=test_pairs,
                test_clusters={k: root(k) for k in parent}, near_duplicate_threshold=threshold), unique_train


def soap_json(text):
    """Convert only explicit SOAP headings; never invent reference sections."""
    try:
        value = json.loads(text)
    except (ValueError, TypeError):
        value = None
    if isinstance(value, dict) and set(value) == set(FIELDS) and all(isinstance(value[k], str) and value[k].strip() for k in FIELDS):
        return value
    matches = list(re.finditer(r'^\s*(Subjective|Objective|Assessment|Plan)\s*:\s*', text, re.I | re.M))
    if len(matches) != 4 or {m.group(1).lower() for m in matches} != set(FIELDS):
        raise ValueError('Reference does not contain exactly four explicit SOAP sections')
    result = {m.group(1).lower(): text[m.end():matches[i + 1].start() if i + 1 < len(matches) else len(text)].strip()
              for i, m in enumerate(matches)}
    if not all(result.values()):
        raise ValueError('Empty reference section')
    return result


def structure(text):
    try:
        value = json.loads(text)
    except (ValueError, TypeError):
        return dict(json_valid=0, schema_valid=0, structural_coverage=0.0)
    valid = isinstance(value, dict) and set(value) == set(FIELDS) and all(isinstance(v, str) and v.strip() for v in value.values())
    coverage = sum(isinstance(value.get(k), str) and bool(value[k].strip()) for k in FIELDS) / 4 if isinstance(value, dict) else 0.0
    return dict(json_valid=1, schema_valid=int(valid), structural_coverage=coverage)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=True, allow_nan=False)


def read_jsonl(path):
    with Path(path).open(encoding='utf-8') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=True, allow_nan=False) + '\n')


def paired_rows(rows, metric, conditions=CONDITIONS[:4]):
    mapped = {}
    for row in rows:
        if row['condition'] not in conditions:
            continue
        key = (row['case_id'], row['condition'])
        if key in mapped:
            raise ValueError(f'Duplicate case/condition: {key}; analyze seeds separately')
        value = row.get(metric)
        if value is None or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f'Missing or invalid {metric} for {key}')
        mapped[key] = value
    ids = sorted({k[0] for k in mapped})
    if not ids or len(mapped) != len(ids) * len(conditions):
        raise ValueError('Incomplete paired condition matrix')
    return ids, [[mapped[(case, condition)] for condition in conditions] for case in ids]


def annotation_packet(rows, test_ids, seed=3407):
    selected = random.Random(seed).sample(sorted(test_ids), min(20, len(test_ids)))
    records = [r for r in rows if r['case_id'] in selected and r['condition'] in CONDITIONS[:4]]
    if any(r.get('status', 'ok') != 'ok' for r in records):
        raise ValueError('Resolve infrastructure failures before clinical annotation')
    paired_rows([dict(r, present=1) for r in records], 'present')
    if len(records) != len(selected) * 4:
        raise ValueError('Selected cases have incomplete generations')
    random.Random(seed).shuffle(records)
    packet, key = [], []
    for i, row in enumerate(records):
        note_id = f'N{i + 1:03d}'
        packet.append(dict(note_id=note_id, conversation=row['conversation'], note=row['output'],
                           unsupported_claims=None, important_omissions=None, section_placement=None,
                           severity=None, acceptable=None, rationale=''))
        key.append(dict(note_id=note_id, case_id=row['case_id'], condition=row['condition']))
    return packet, key
