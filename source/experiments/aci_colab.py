"""Run only the GPU-heavy ACI steps (LoRA training seed 3407, LoRA-only and RAG+LoRA generation)
on a cloud GPU, under the frozen ACI protocol. Judging and the summary stay on the local machine:
copy revision/results/aci/adapters/seed3407 and the two generation files back, then rerun
run_aci_replication.bat (finished steps are skipped).

    python -m source.experiments.aci_colab
"""
from .aci_replication import OUT, CONFIG, SEED, log, step
from .revision_runner import generate, load_config, train


def main():
    config, audit = load_config(CONFIG)
    adapter = OUT / f'adapters/seed{SEED}'
    ok = step(f'train LoRA seed {SEED} (uncapped, frozen protocol)', adapter / 'revision_manifest.json',
              lambda: train(config, audit, adapter, SEED))
    for cond, ctx in (('lora', None), ('lora_rag', OUT / 'contexts.jsonl')):
        gen = OUT / f'generations_{cond}.jsonl'
        ok &= step(f'generate {cond}', gen, lambda: generate(config, gen, ctx, cond, adapter, SEED))
    log('DONE' if ok else 'FAILED')


if __name__ == '__main__':
    main()
