"""Score SOAP notes with TypeSafe Jev (System One, Noul questions).

Design follows the TypeSafe docs (docs.typesafe.ai, primitives/noul):
- one specific yes/no question per Noul, phrased so that a high value means "yes";
- explicit `criteria` for the yes/no boundary;
- Noul values are probabilities of "yes", not degrees: report bands with 0.7/0.3 cut-offs
  (values in between are "indeterminate" and are candidates for human review);
- questions in one request are evaluated independently against the same state.

Question sets
- v1: the two compound questions used for revision/results/jev_*_v1.jsonl (kept for reproducibility).
- v2 (default): faithfulness decomposed into six single-scope defect checks plus relevance.
  A note is "supported" when no defect check is likely (max defect probability < 0.5);
  the band uses 0.3/0.7 on the same maximum.

Usage (run from the repository root; needs TYPE_SAFE_JEV_API_KEY or TYPESAFE_API_KEY):
    python -m source.experiments.jev_judge run --inputs revision/results/rescore_v1/inputs --out revision/results/jev_v2 --model jev-1.14.0
    python -m source.experiments.jev_judge score --input FILE.jsonl --output OUT.jsonl
Runs are resumable: finished cases are kept in <output>.partial and skipped on rerun.
"""
import argparse
import hashlib
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

ENDPOINT = 'https://api.typesafe.ai/v1/systemone'
DEFAULT_MODEL = 'jev-latest'
YES, NO = 0.7, 0.3
GROUPS = ('base', 'rag', 'lora', 'lora_rag', 'random_rag', 'schema', 'reference_test', 'reference_train')


def noul(instructions, true, false):
    return {'type': 'noul', 'instructions': instructions, 'criteria': {'true': true, 'false': false}}


QUESTION_SETS = {
    'v1': {
        'faithfulness': {'type': 'noul', 'instructions': (
            'Is every substantive clinical claim in the SOAP note supported by the current conversation, '
            'with no invented findings, diagnoses, measurements, or treatments?')},
        'answer_relevancy': {'type': 'noul', 'instructions': (
            'Does the SOAP note address the current clinical encounter and remain relevant to it '
            'without major off-topic content?')},
    },
    'v2': {
        'unsupported_measurement': noul(
            'Does the SOAP note state a vital sign, laboratory value, test result, weight, or medication dose '
            'that is not given in the conversation?',
            'At least one such value in the note is absent from the conversation.',
            'Every such value in the note appears in the conversation, or the note states none.'),
        'unsupported_exam_finding': noul(
            'Does the SOAP note report a physical examination finding or observation that the conversation '
            'does not describe?',
            'The note reports an examination finding that the conversation does not mention.',
            'All examination findings in the note are described in the conversation, or the note reports none.'),
        'unsupported_diagnosis': noul(
            'Does the SOAP note state a diagnosis or condition that the physician in the conversation did not '
            'state or clearly discuss?',
            'The note asserts a diagnosis or condition not stated or discussed in the conversation.',
            'Every diagnosis or condition in the note is stated or discussed in the conversation.'),
        'unsupported_plan': noul(
            'Does the SOAP note include a medication, test, procedure, referral, or follow-up that was not '
            'discussed in the conversation?',
            'The plan contains at least one item that the conversation did not discuss.',
            'Every plan item in the note was discussed in the conversation.'),
        'contradicts_conversation': noul(
            'Does the SOAP note contradict something the patient or physician said in the conversation?',
            'The note states something the conversation contradicts, for example a symptom the patient denied.',
            'Nothing in the note contradicts the conversation.'),
        'other_patient_content': noul(
            'Does the SOAP note contain a name, symptom, history, or event that belongs to a different patient '
            'or encounter than the one in the conversation?',
            'The note contains details that do not belong to this patient or encounter.',
            'All details in the note concern this patient and encounter.'),
        'relevant': noul(
            'Does the SOAP note document the clinical encounter in the conversation?',
            'The note is about this encounter and its main problems.',
            'The note is mostly about something else or misses the encounter entirely.'),
    },
}
DEFECTS = ('unsupported_measurement', 'unsupported_exam_finding', 'unsupported_diagnosis',
           'unsupported_plan', 'contradicts_conversation', 'other_patient_content')


def digest(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def band(p):
    return 'yes' if p >= YES else 'no' if p <= NO else 'indeterminate'


def read_jsonl(path):
    with Path(path).open(encoding='utf-8') as handle:
        return [json.loads(line) for line in handle if line.strip()]


def api_key():
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    key = os.environ.get('TYPE_SAFE_JEV_API_KEY') or os.environ.get('TYPESAFE_API_KEY')
    if not key:
        raise RuntimeError('TYPE_SAFE_JEV_API_KEY or TYPESAFE_API_KEY is required')
    return key


def make_state(row, question_set):
    if question_set == 'v1':  # exact v1 state string, for reproducibility
        return 'CURRENT CONVERSATION:\n' + row['conversation'] + '\n\nSOAP NOTE TO EVALUATE:\n' + row['output']
    try:
        note = json.loads(row['output'])
    except (TypeError, ValueError):
        note = row['output']
    return {'conversation': row['conversation'], 'soap_note': note}


def post(api_key_value, payload, timeout=90):
    request = Request(ENDPOINT, data=json.dumps(payload).encode('utf-8'), method='POST', headers={
        'Authorization': f'Bearer {api_key_value}', 'Content-Type': 'application/json', 'Accept': 'application/json'})
    with urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode('utf-8'))


def interpret(answers, question_set, threshold=0.5):
    """Turn raw Noul answers into the fields used downstream."""
    probs = {}
    for qid in QUESTION_SETS[question_set]:
        value = float(answers[qid]['noul'])
        if not 0.0 <= value <= 1.0:
            raise ValueError(f'Noul value outside [0, 1] for {qid}')
        probs[qid] = value
    if question_set == 'v1':
        faith, relevance = probs['faithfulness'], probs['answer_relevancy']
        return dict(probabilities=probs, faithfulness=int(faith >= threshold), answer_relevancy=int(relevance >= threshold),
                    faithfulness_probability=faith, answer_relevancy_probability=relevance)
    worst = max(probs[q] for q in DEFECTS)
    return dict(probabilities=probs,
                defects={q: band(probs[q]) for q in DEFECTS},
                max_defect_probability=worst,
                faithfulness=int(worst < 0.5),
                faithfulness_band={'yes': 'unsupported', 'no': 'supported', 'indeterminate': 'indeterminate'}[band(worst)],
                answer_relevancy=int(probs['relevant'] >= 0.5),
                relevance_band=band(probs['relevant']))


def score_row(row, key, question_set, model, transport=post, retries=3):
    entry = dict(row, judge_provider='TypeSafe', judge_endpoint=ENDPOINT, judge_model=model,
                 judge_question_set=question_set, judge_status='missing', faithfulness=None,
                 answer_relevancy=None, judge_attempts=[])
    if row.get('status', 'ok') != 'ok':
        entry['judge_status'] = 'not_scored_generation_failure'
        return entry
    if not str(row.get('output', '')).strip():
        entry.update(faithfulness=0, answer_relevancy=0, judge_status='deterministic_empty_output')
        return entry
    payload = {'state': make_state(row, question_set), 'model': model, 'questions': QUESTION_SETS[question_set]}
    entry['judge_prompt_hash'] = digest(json.dumps(payload, sort_keys=True))
    for attempt in range(retries):
        try:
            response = transport(key, payload)
            entry.update(interpret(response['answers'], question_set), judge_status='ok',
                         judge_returned=response.get('model'), judge_usage=response.get('usage'), judge_raw=response)
            entry['judge_attempts'].append({'status': 'ok'})
            return entry
        except HTTPError as exc:
            entry['judge_attempts'].append({'error': f'HTTP {exc.code}: {exc.reason}'})
            if exc.code in (400, 401, 403, 404, 422):  # not retryable
                return entry
        except (URLError, TimeoutError, KeyError, ValueError, json.JSONDecodeError) as exc:
            entry['judge_attempts'].append({'error': f'{type(exc).__name__}: {exc}'})
        time.sleep(2 ** attempt)
    return entry


def score(input_path, output_path, question_set='v2', model=DEFAULT_MODEL, workers=4, transport=post, key=None):
    """Score one JSONL file. Resumable through <output>.partial; the final file is written exclusively."""
    output_path = Path(output_path)
    if output_path.exists():
        raise ValueError(f'Refusing to overwrite existing output: {output_path}')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    partial = output_path.with_name(output_path.name + '.partial')
    rows = read_jsonl(input_path)
    if len({r['case_id'] for r in rows}) != len(rows):
        raise ValueError(f'Duplicate case_id in {input_path}')
    done = {r['case_id']: r for r in read_jsonl(partial) if r.get('judge_status') != 'missing'} if partial.exists() else {}
    todo = [r for r in rows if r['case_id'] not in done]
    key = key or api_key()
    lock = threading.Lock()
    with partial.open('w', encoding='utf-8', newline='\n') as handle:
        for r in done.values():
            handle.write(json.dumps(r, ensure_ascii=True) + '\n')
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(score_row, r, key, question_set, model, transport) for r in todo]
            for i, future in enumerate(as_completed(futures), 1):
                entry = future.result()
                with lock:
                    handle.write(json.dumps(entry, ensure_ascii=True) + '\n')
                    handle.flush()
                    if entry['judge_status'] == 'ok':
                        done[entry['case_id']] = entry
                if i % 25 == 0:
                    print(f'  {output_path.stem}: {i}/{len(todo)}', flush=True)
    missing = [r['case_id'] for r in rows if r['case_id'] not in done]
    if missing:
        print(f'  {output_path.stem}: {len(missing)} missing judgments; rerun the same command to retry them', flush=True)
        return False
    ordered = [done[r['case_id']] for r in rows]
    with output_path.open('x', encoding='utf-8', newline='\n') as handle:
        for entry in ordered:
            handle.write(json.dumps(entry, ensure_ascii=True) + '\n')
    partial.unlink()
    return True


def summarize(out_dir):
    out_dir = Path(out_dir)
    summary = {'question_set': None, 'groups': []}
    for group in GROUPS:
        path = out_dir / f'{group}.jsonl'
        if not path.exists():
            continue
        rows = read_jsonl(path)
        ok = [r for r in rows if r['judge_status'] in ('ok', 'deterministic_empty_output')]
        summary['question_set'] = rows[0]['judge_question_set']
        entry = dict(group=group, n=len(rows), scored=len(ok),
                     judge_returned=sorted({r.get('judge_returned') for r in ok if r.get('judge_returned')}),
                     supported=sum(r['faithfulness'] for r in ok), relevant=sum(r['answer_relevancy'] for r in ok))
        if summary['question_set'] == 'v2':
            entry['faithfulness_bands'] = {b: sum(r.get('faithfulness_band') == b for r in ok)
                                           for b in ('supported', 'indeterminate', 'unsupported')}
            entry['defect_likely'] = {q: sum(r.get('defects', {}).get(q) == 'yes' for r in ok) for q in DEFECTS}
        summary['groups'].append(entry)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='action', required=True)
    one = sub.add_parser('score', help='score one JSONL file')
    one.add_argument('--input', type=Path, required=True)
    one.add_argument('--output', type=Path, required=True)
    many = sub.add_parser('run', help='score every condition and reference file in an inputs folder')
    many.add_argument('--inputs', type=Path, required=True)
    many.add_argument('--out', type=Path, required=True)
    for p in (one, many):
        p.add_argument('--question-set', choices=sorted(QUESTION_SETS), default='v2')
        p.add_argument('--model', default=DEFAULT_MODEL, help='pin a release, e.g. jev-1.14.0')
        p.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    if args.action == 'score':
        ok = score(args.input, args.output, args.question_set, args.model, args.workers)
    else:
        ok = True
        for group in GROUPS:
            source, target = args.inputs / f'{group}.jsonl', args.out / f'{group}.jsonl'
            if source.exists() and not target.exists():
                print(f'Jev {args.question_set} on {group} ...', flush=True)
                ok = score(source, target, args.question_set, args.model, args.workers) and ok
        if ok:
            target = args.out / 'summary.json'
            target.write_text(json.dumps(summarize(args.out), indent=2), encoding='utf-8')
            print(f'wrote {target}')
    raise SystemExit(0 if ok else 1)


if __name__ == '__main__':
    main()
