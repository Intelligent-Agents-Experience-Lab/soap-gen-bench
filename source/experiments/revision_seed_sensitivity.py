"""LoRA seed sensitivity (seeds 3408, 3409): train, generate, score with every measure, summarize.

Run from the repository root on the GPU machine (resumable; every finished step is skipped):
    python -m source.experiments.revision_seed_sensitivity

Steps per seed: train adapter -> generate LoRA-only and RAG+LoRA on the 62 test cases
(same frozen protocol and cached contexts as seed 3407) -> score with gpt-4o-mini-2024-07-18
(prespecified), gpt-5.6-luna (medium reasoning), Jev v2 (jev-1.14.0) and the numeric
measurement audit -> paired comparisons with the seed-3407 Base arm. Seeds are analysed
separately and never pooled as extra cases.
"""
import argparse
import json
import traceback
from pathlib import Path

from scipy.stats import binomtest

from .revision_core import read_jsonl, write_json
from .revision_runner import generate, judge, load_config, train

R = Path('revision/results')
OUT = R / 'seeds'
SEEDS = (3408, 3409)
CONDS = ('lora', 'lora_rag')
CONTEXTS = R / 'contexts_v5.jsonl'
LUNA = 'gpt-5.6-luna'
BASE = {'gpt-4o-mini (primary)': R / 'rescore_v1/scored/{c}.jsonl',
        'gpt-5.6-luna': R / 'rescore_luna_v1/scored/{c}.jsonl',
        'jev v2': R / 'jev_v2/{c}.jsonl'}


def log(msg):
    print(msg, flush=True)


def step(name, target, fn):
    if Path(target).exists():
        log(f'[skip] {name}: {target} exists')
        return True
    log(f'[run ] {name} -> {target}')
    try:
        fn()
        return True
    except Exception:
        log(f'[FAIL] {name}\n{traceback.format_exc()}')
        return False
    finally:
        free_gpu()


def free_gpu():
    """Release the previous step's model so the next load gets the whole GPU."""
    import gc
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except ImportError:
        pass


def audit_rows(rows):
    from .revision_measurement_audit import unsupported
    return {r['case_id']: int(bool(unsupported(r['output'], r['conversation']))) for r in rows}


def mcnemar(base, arm):
    pos = sum(1 for k in base if base[k] == 0 and arm.get(k) == 1)
    neg = sum(1 for k in base if base[k] == 1 and arm.get(k) == 0)
    return dict(discordant=[pos, neg], p=float(binomtest(pos, pos + neg).pvalue) if pos + neg else 1.0)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--seeds', type=int, nargs='+', default=list(SEEDS))
    parser.add_argument('--skip-judges', action='store_true', help='only train and generate')
    args = parser.parse_args()
    config, audit = load_config(Path('revision/protocol/config.json'))
    OUT.mkdir(parents=True, exist_ok=True)
    ok = True
    for seed in args.seeds:
        adapter = R / f'adapters/seed{seed}'
        if not step(f'train seed {seed}', adapter / 'revision_manifest.json',
                    lambda: train(config, audit, adapter, seed)):
            ok = False
            continue
        for cond in CONDS:
            gen = OUT / f'generations_{cond}_s{seed}.jsonl'
            ok &= step(f'generate {cond} seed {seed}', gen,
                       lambda: generate(config, gen, CONTEXTS if cond == 'lora_rag' else None, cond, adapter, seed))
            if args.skip_judges or not gen.exists():
                continue
            ok &= step(f'gpt-4o-mini {cond} s{seed}', OUT / f'gpt4omini_{cond}_s{seed}.jsonl',
                       lambda: judge(config, gen, OUT / f'gpt4omini_{cond}_s{seed}.jsonl'))
            ok &= step(f'luna {cond} s{seed}', OUT / f'luna_{cond}_s{seed}.jsonl',
                       lambda: judge(dict(config, judge=LUNA), gen, OUT / f'luna_{cond}_s{seed}.jsonl',
                                     reasoning_effort='medium'))
            from .jev_judge import score as jev_score
            ok &= step(f'jev v2 {cond} s{seed}', OUT / f'jevv2_{cond}_s{seed}.jsonl',
                       lambda: jev_score(gen, OUT / f'jevv2_{cond}_s{seed}.jsonl', 'v2', 'jev-1.14.0', 4) or
                       (_ for _ in ()).throw(RuntimeError('Jev v2 left missing judgments; rerun to retry')))
    # summary (whatever is available)
    summary = dict(note='Seeds analysed separately; Base arm is the seed-3407 run (no adapter, seed-independent).',
                   seeds={})
    prefix = {'gpt-4o-mini (primary)': 'gpt4omini', 'gpt-5.6-luna': 'luna', 'jev v2': 'jevv2'}
    base_meas = audit_rows(read_jsonl(R / 'rescore_v1/inputs/base.jsonl'))
    for seed in [3407] + list(args.seeds):
        s = {}
        for cond in CONDS:
            c = {}
            for j, pat in BASE.items():
                path = Path(str(pat).format(c=cond)) if seed == 3407 else OUT / f'{prefix[j]}_{cond}_s{seed}.jsonl'
                if not path.exists():
                    continue
                rows = {r['case_id']: r['faithfulness'] for r in read_jsonl(path)}
                if any(v is None for v in rows.values()):
                    c[j] = dict(status='missing judgments', missing=sum(v is None for v in rows.values()))
                    continue
                base = {r['case_id']: r['faithfulness'] for r in read_jsonl(Path(str(pat).format(c='base')))}
                c[j] = dict(supported=sum(rows.values()), n=len(rows), base_supported=sum(base.values()),
                            vs_base=mcnemar(base, rows))
            gen = (R / 'rescore_v1/inputs' / f'{cond}.jsonl') if seed == 3407 else OUT / f'generations_{cond}_s{seed}.jsonl'
            if gen.exists():
                grows = read_jsonl(gen)
                meas = audit_rows(grows)
                c['measurement audit'] = dict(notes_with_unsupported_measurement=sum(meas.values()),
                                              base=sum(base_meas.values()), vs_base=mcnemar(base_meas, meas))
                c['structurally_valid'] = sum(int(bool(r.get('schema_valid'))) for r in grows)
                c['adapter_training_loss'] = (json.loads((R / f'adapters/seed{seed}/revision_manifest.json')
                                                         .read_text(encoding='utf-8')).get('training_loss'))
            s[cond] = c
        summary['seeds'][str(seed)] = s
    target = OUT / 'summary.json'
    target.write_text(json.dumps(summary, indent=2), encoding='utf-8')
    log(f'wrote {target}')
    log('ALL STEPS DONE' if ok else 'SOME STEPS FAILED - rerun the same command to retry')
    raise SystemExit(0 if ok else 1)


if __name__ == '__main__':
    main()
