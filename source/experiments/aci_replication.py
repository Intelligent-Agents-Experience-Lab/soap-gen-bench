"""ACI-BENCH replication of the held-out factorial evaluation (resumable; run on the GPU machine).

    python -m source.experiments.aci_replication            # everything
    python -m source.experiments.aci_replication --skip-judges

Every step writes one output and is skipped when that output exists, so rerunning the same
command resumes after a crash. Steps:

 1. prepare   data/external/aci-bench -> revision/results/data_aci_v1 (train+valid = 87, test = 120)
 2. freeze    token budgets from the TRAINING split only (tokenizer), a training smoke run on the
              8 longest training examples (memory check) and a retrieval smoke run; then the config
              revision/protocol/config_aci.json is marked FROZEN with a freeze record
 3. reference control: the 120 ACI test reference notes scored against their own conversations
              by gpt-4o-mini-2024-07-18, gpt-5.6-luna and Jev v2, plus the numeric audit v2
 4. retrieve  hybrid BM25 + MiniLM retrieval over the 87 training notes, k = 3, cached once
 5. train     LoRA seed 3407 on the training split (same recipe as the main experiment)
 6. generate  base, rag, lora, lora_rag, random_rag, schema on the 120 test cases
 7. judge     each arm with the three judges; numeric audit v2
 8. summary   revision/results/aci/summary.json
"""
import argparse
import datetime
import json
import math
import traceback
from pathlib import Path

from .revision_core import CONDITIONS, read_jsonl, write_json, write_jsonl

CONFIG = Path('revision/protocol/config_aci.json')
DATA = Path('revision/results/data_aci_v1')
OUT = Path('revision/results/aci')
SEED = 3407
# Deviation recorded 2026-11-01, before any LoRA output existed: on the 12 GB RTX 3060 the longest
# training examples spilled into system memory (step time 12-38 min, ~35 h projected). Training examples
# longer than this cap are excluded, the same rule the main experiment applied at its own limit.
TRAIN_TOKEN_CAP = 3584
LUNA = 'gpt-5.6-luna'
JEV = 'jev-1.14.0'
JUDGES = {'gpt4omini': 'gpt-4o-mini-2024-07-18 (prespecified)', 'luna': 'gpt-5.6-luna (post hoc)',
          'jevv2': 'Jev v2 jev-1.14.0 (post hoc)'}


def log(msg):
    print(msg, flush=True)


def free_gpu():
    import gc
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except ImportError:
        pass


def step(name, target, fn):
    if Path(target).exists():
        log(f'[skip] {name}: {target} exists')
        return True
    log(f'[run ] {name} -> {target}  ({datetime.datetime.now():%H:%M})')
    try:
        fn()
        return True
    except Exception:
        log(f'[FAIL] {name}\n{traceback.format_exc()}')
        return False
    finally:
        free_gpu()


def roundup(value, unit):
    return int(math.ceil(value / unit) * unit)


def budgets(config, train_rows, sft_rows, count_prompt, count_target):
    """Token budgets derived from the training split only."""
    prompt_tokens = [count_prompt(r['conversation']) for r in train_rows]
    target_tokens = {r['case_id']: count_target(r['target']) for r in sft_rows}
    by_id = dict(zip([r['case_id'] for r in train_rows], prompt_tokens))
    max_new = roundup(max(target_tokens.values()) * 1.1, 128)
    sft_tokens = roundup(max(by_id[k] + v for k, v in target_tokens.items()), 256)
    needed = max(prompt_tokens) + config['context_tokens'] + max_new
    if needed > config['max_sequence_tokens']:
        raise ValueError(f'Training prompts need {needed} tokens > max_sequence_tokens {config["max_sequence_tokens"]}')
    return dict(max_new_tokens=max_new, sft_tokens=sft_tokens, max_training_input_tokens=max(prompt_tokens),
                max_training_target_tokens=max(target_tokens.values()), required_sequence_tokens=needed)


def freeze():
    from transformers import AutoTokenizer
    from .revision_runner import load_config, prompt, retrieve, train, versions
    config, audit = load_config(CONFIG, require_frozen=False)
    tok = AutoTokenizer.from_pretrained(config['model'], revision=config['model_revision'], token=False)
    count_prompt = lambda text: len(tok.apply_chat_template([dict(role='user', content=prompt(text))],
                                                            tokenize=True, add_generation_prompt=True))
    count_target = lambda target: len(tok.encode(json.dumps(target), add_special_tokens=False)) + 1
    b = budgets(config, read_jsonl(DATA / 'train.jsonl'), read_jsonl(DATA / 'sft.jsonl'), count_prompt, count_target)
    log(f'       budgets from training split: {b}')
    config.update(max_new_tokens=b['max_new_tokens'], sft_tokens=b['sft_tokens'])
    smoke_adapter, smoke_retrieval = OUT / 'adapters/smoke', OUT / 'retrieval_smoke.jsonl'
    if not (smoke_adapter / 'revision_manifest.json').exists():
        train(config, audit, smoke_adapter, SEED, smoke=True)   # 8 longest examples: memory check
    free_gpu()
    if not smoke_retrieval.exists():
        retrieve(config, smoke_retrieval, smoke=True)
    stored = json.loads(CONFIG.read_text(encoding='utf-8'))
    stored.update(status='FROZEN', max_new_tokens=b['max_new_tokens'], sft_tokens=b['sft_tokens'],
                  token_budget_status='VERIFIED_ON_TRAIN',
                  freeze_record=dict(date=datetime.date.today().isoformat(), **b,
                                     training_smoke=str(smoke_adapter / 'revision_manifest.json').replace('\\', '/'),
                                     retrieval_smoke=str(smoke_retrieval).replace('\\', '/'),
                                     runtime_packages=versions()))
    CONFIG.write_text(json.dumps(stored, indent=2) + '\n', encoding='utf-8')
    write_json(OUT / 'freeze.json', stored['freeze_record'])


def reference_rows():
    rows = []
    for r in read_jsonl(DATA / 'test.jsonl'):
        note = json.dumps(r['target']) if r['target'] else r['reference_original']
        rows.append(dict(case_id=r['case_id'], conversation=r['conversation'], condition='reference',
                         status='ok', output=note, subset=r['subset']))
    return rows


def judge_all(config, name, gen, ok):
    from .jev_judge import score as jev_score
    from .revision_runner import judge

    def jev():
        jev_score(gen, OUT / f'jevv2_{name}.jsonl', 'v2', JEV, 4)
        if any(r['faithfulness'] is None for r in read_jsonl(OUT / f'jevv2_{name}.jsonl')):
            raise RuntimeError('Jev v2 left missing judgments; delete the file and rerun to retry them')
    ok &= step(f'gpt-4o-mini {name}', OUT / f'gpt4omini_{name}.jsonl',
               lambda: judge(config, gen, OUT / f'gpt4omini_{name}.jsonl'))
    ok &= step(f'luna {name}', OUT / f'luna_{name}.jsonl',
               lambda: judge(dict(config, judge=LUNA), gen, OUT / f'luna_{name}.jsonl', reasoning_effort='medium'))
    ok &= step(f'jev v2 {name}', OUT / f'jevv2_{name}.jsonl', jev)
    return ok


def flags(rows):
    from .revision_measurement_audit_v2 import unsupported
    return {r['case_id']: int(bool(unsupported(r['output'], r['conversation']))) for r in rows}


def mcnemar(left, right):
    from scipy.stats import binomtest
    pos = sum(1 for k in left if left[k] == 0 and right.get(k) == 1)
    neg = sum(1 for k in left if left[k] == 1 and right.get(k) == 0)
    return dict(discordant=[pos, neg], p=float(binomtest(pos, pos + neg).pvalue) if pos + neg else 1.0,
                test='exact McNemar')


def summarize(audit):
    from .revision_statistics import analyze
    clusters = audit['test_clusters']
    s = dict(dataset='ACI-BENCH', n_test=audit['test_count'], n_train=audit['retained_train_count'],
             sft_count=audit['sft_count'], judges={}, measurement_audit_v2={}, structure={})
    for prefix, label in JUDGES.items():
        j = {}
        for name in ('reference',) + CONDITIONS:
            path = OUT / f'{prefix}_{name}.jsonl'
            if path.exists():
                rows = read_jsonl(path)
                missing = sum(r['faithfulness'] is None for r in rows)
                j[name] = dict(supported=sum(r['faithfulness'] or 0 for r in rows), n=len(rows), missing=missing)
                if prefix == 'jevv2' and rows and all(r.get('defects') for r in rows):
                    j[name]['defects_likely'] = {q: sum(r['defects'][q] == 'yes' for r in rows) for q in rows[0]['defects']}
        primary = [OUT / f'{prefix}_{c}.jsonl' for c in ('base', 'rag', 'lora', 'lora_rag')]
        if all(p.exists() for p in primary):
            rows = [dict(r, condition=c) for c, p in zip(('base', 'rag', 'lora', 'lora_rag'), primary) for r in read_jsonl(p)]
            if all(r['faithfulness'] is not None for r in rows):
                j['primary_contrasts'] = analyze(rows, 'faithfulness', clusters, adjust_interaction=False)
        for arm, ref in (('random_rag', 'rag'), ('schema', 'base')):
            a, b = OUT / f'{prefix}_{arm}.jsonl', OUT / f'{prefix}_{ref}.jsonl'
            if a.exists() and b.exists():
                j[f'{arm}_vs_{ref}'] = mcnemar({r['case_id']: r['faithfulness'] for r in read_jsonl(b)},
                                               {r['case_id']: r['faithfulness'] for r in read_jsonl(a)})
        s['judges'][label] = j
    base = None
    for name in ('reference',) + CONDITIONS:
        path = OUT / ('reference_inputs.jsonl' if name == 'reference' else f'generations_{name}.jsonl')
        if not path.exists():
            continue
        rows = read_jsonl(path)
        f = flags(rows)
        base = f if name == 'base' else base
        entry = dict(notes_with_unsupported_measurement=sum(f.values()), n=len(f))
        if name not in ('reference', 'base') and base:
            entry['vs_base'] = mcnemar(base, f)
        s['measurement_audit_v2'][name] = entry
        if name != 'reference':
            s['structure'][name] = dict(schema_valid=sum(int(bool(r.get('schema_valid'))) for r in rows),
                                        hit_output_limit=sum(int(bool(r.get('hit_output_limit'))) for r in rows),
                                        failures=sum(r.get('status') != 'ok' for r in rows), n=len(rows))
    manifest = OUT / f'adapters/seed{SEED}/revision_manifest.json'
    if manifest.exists():
        m = json.loads(manifest.read_text(encoding='utf-8'))
        s['adapter'] = dict(seed=m['seed'], training_loss=m['training_loss'], eligible=m['eligible_count'],
                            token_exclusions=m['token_exclusions'], global_step=m['global_step'])
    return s


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--skip-judges', action='store_true', help='only prepare, train and generate')
    parser.add_argument('--judge-only', action='store_true', help='only judge generation files that already exist, then summarize')
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    from .revision_runner import generate, load_config, retrieve, train
    from .aci_prepare import main as prepare
    ok = step('prepare ACI-BENCH', DATA / 'audit.json', lambda: prepare([]))
    if not ok or not step('freeze budgets on training split', OUT / 'freeze.json', freeze):
        log('STOPPED before held-out work - fix the error above and rerun')
        raise SystemExit(1)
    config, audit = load_config(CONFIG)
    if not args.skip_judges:
        ok &= step('reference inputs', OUT / 'reference_inputs.jsonl',
                   lambda: write_jsonl(OUT / 'reference_inputs.jsonl', reference_rows()))
        ok = judge_all(config, 'reference', OUT / 'reference_inputs.jsonl', ok)
    if args.judge_only:
        for cond in CONDITIONS:
            gen = OUT / f'generations_{cond}.jsonl'
            if gen.exists():
                ok = judge_all(config, cond, gen, ok)
        (OUT / 'summary.json').write_text(json.dumps(summarize(audit), indent=2), encoding='utf-8')
        log('JUDGE-ONLY DONE' if ok else 'SOME JUDGE STEPS FAILED - rerun to retry')
        raise SystemExit(0 if ok else 1)
    contexts = OUT / 'contexts.jsonl'
    ok &= step('retrieve (k = 3)', contexts, lambda: retrieve(config, contexts))
    adapter = OUT / f'adapters/seed{SEED}'
    if adapter.exists() and not (adapter / 'revision_manifest.json').exists() and not any(adapter.rglob('*.safetensors')):
        import shutil
        shutil.rmtree(adapter)  # empty folder left by an interrupted training run
        log(f'[note] removed incomplete {adapter}')
    ok &= step(f'train LoRA seed {SEED}', adapter / 'revision_manifest.json', lambda: train(config, audit, adapter, SEED, token_limit=TRAIN_TOKEN_CAP))
    for cond in CONDITIONS:
        gen = OUT / f'generations_{cond}.jsonl'
        uses_lora, uses_context = cond in ('lora', 'lora_rag'), cond in ('rag', 'lora_rag', 'random_rag')
        if (uses_lora and not adapter.exists()) or (uses_context and not contexts.exists()):
            log(f'[wait] generate {cond}: needs the adapter/contexts from a failed step')
            continue
        ok &= step(f'generate {cond}', gen, lambda: generate(config, gen, contexts if uses_context else None,
                                                            cond, adapter if uses_lora else None, SEED))
        if not args.skip_judges and gen.exists():
            ok = judge_all(config, cond, gen, ok)
    target = OUT / 'summary.json'
    target.write_text(json.dumps(summarize(audit), indent=2), encoding='utf-8')
    log(f'wrote {target}')
    log('ALL STEPS DONE' if ok else 'SOME STEPS FAILED - rerun the same command to retry')
    raise SystemExit(0 if ok else 1)


if __name__ == '__main__':
    main()
