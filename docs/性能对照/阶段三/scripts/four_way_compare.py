"""Measure four pinned engines with the same step-driven workload and clock.

Run in separate processes; select source tree through PYTHONPATH. The driver
observes schedule/step results but does not modify inference code.
"""
import argparse
import atexit
import hashlib
import json
import random
import statistics
import time

import torch


MODEL = '/home/ubuntu/ai-infra/minivllm/Models/Qwen3-0.6B'
CONFIG = dict(max_num_batched_tokens=512, max_num_seqs=8, max_model_len=2048,
              gpu_memory_utilization=.85, enforce_eager=False, tensor_parallel_size=1)
CASES = {
    'short_batch': [(128, 32)] * 4,
    'long_batch': [(1024, 16)] * 4,
    'staggered': [(64, 64), (1024, 16), (1024, 16)],
}


def percentile(xs, p):
    xs = sorted(xs)
    pos = (len(xs) - 1) * p
    lo = int(pos)
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


def workload(engine, SamplingParams, flavor, case, repeat):
    rng = random.Random(810000 + repeat * 11 + list(CASES).index(case))
    prompts = [[rng.randint(1, 10000) for _ in range(length)] for length, _ in CASES[case]]
    torch.manual_seed(20261028 + repeat)
    torch.cuda.manual_seed_all(20261028 + repeat)
    records = []
    observed = {'mixed_steps': 0, 'prefill_steps': 0, 'decode_steps': 0, 'steps': 0, 'target_forwards': 0, 'verification_forwards': 0, 'speculative_items': 0}
    original_schedule = engine.scheduler.schedule

    def traced_schedule():
        output = original_schedule()
        if flavor == 'nano':
            seqs, prefill = output
            observed['prefill_steps' if prefill else 'decode_steps'] += 1
        else:
            prefill, decode = output.has_prefill, output.has_decode
            observed['prefill_steps'] += int(prefill)
            observed['decode_steps'] += int(decode)
            observed['mixed_steps'] += int(prefill and decode)
            observed['speculative_items'] += sum(item.mode.name == 'SPECULATIVE' for item in output.scheduled)
        observed['steps'] += 1
        return output

    engine.scheduler.schedule = traced_schedule
    original_call = engine.model_runner.call

    def traced_call(name, *args):
        if name in ('run', 'run_speculative'):
            observed['target_forwards'] += 1
        if name == 'run_speculative':
            observed['verification_forwards'] += 1
        return original_call(name, *args)

    engine.model_runner.call = traced_call

    def submit(i):
        start = time.perf_counter()
        params = SamplingParams(temperature=.6, max_tokens=CASES[case][i][1], ignore_eos=True)
        result = engine.add_request(prompts[i], params)
        seq = engine.scheduler.waiting[-1] if flavor == 'nano' else result
        records.append({'seq': seq, 'arrival': start, 'times': [], 'previous': 0})

    torch.cuda.synchronize()
    start = time.perf_counter()
    submit(0)
    if case != 'staggered':
        for i in range(1, len(prompts)):
            submit(i)
    submitted = case != 'staggered'
    while not engine.is_finished():
        engine.step()
        now = time.perf_counter()
        for rec in records:
            count = rec['seq'].num_completion_tokens
            if count > rec['previous']:
                rec['times'].extend([now] * (count - rec['previous']))
                rec['previous'] = count
        if not submitted and records[0]['previous'] >= 4:
            submit(1)
            submit(2)
            submitted = True
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    engine.scheduler.schedule = original_schedule
    engine.model_runner.call = original_call
    for rec, (_, wanted) in zip(records, CASES[case]):
        if len(rec['times']) != wanted:
            raise RuntimeError(f'incomplete request: {len(rec["times"])} != {wanted}')
    ttft = [r['times'][0] - r['arrival'] for r in records]
    tpot = [(r['times'][-1] - r['times'][0]) / (len(r['times']) - 1) for r in records]
    itl = [b - a for r in records for a, b in zip(r['times'], r['times'][1:])]
    outputs = [r['seq'].completion_token_ids for r in records]
    return dict(case=case, repeat=repeat, elapsed_s=elapsed,
                output_tokens=sum(map(len, outputs)), output_tokens_per_s=sum(map(len, outputs))/elapsed,
                ttft_p50_s=statistics.median(ttft), ttft_p95_s=percentile(ttft, .95),
                tpot_p50_s=statistics.median(tpot), itl_p95_s=percentile(itl, .95),
                e2e_p50_s=statistics.median(r['times'][-1]-r['arrival'] for r in records),
                lead_itl_p95_s=percentile([b-a for a,b in zip(records[0]['times'], records[0]['times'][1:])], .95),
                lead_itl_max_s=max(b-a for a,b in zip(records[0]['times'], records[0]['times'][1:])),
                request_token_offsets_s=[[t-r['arrival'] for t in r['times']] for r in records],
                output_sha256=hashlib.sha256(json.dumps(outputs).encode()).hexdigest(),
                proposed=sum(getattr(r['seq'], 'draft_proposed', 0) for r in records),
                accepted=sum(getattr(r['seq'], 'draft_accepted', 0) for r in records),
                output_bursts=sum(sum(a == b for a,b in zip(r['times'], r['times'][1:])) for r in records),
                **observed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--engine', choices=('nano', 'stage1', 'stage2', 'stage3'), required=True)
    parser.add_argument('--model', default=MODEL)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    if args.engine == 'nano':
        from nanovllm import LLM, SamplingParams
        import nanovllm as package
    else:
        from wwvllm import LLM, SamplingParams
        import wwvllm as package
    options = dict(CONFIG)
    if args.engine in ('stage2', 'stage3'):
        options.update(enable_speculative=True, speculative_model=args.model, speculative_num_tokens=4)
    llm = LLM(args.model, **options)
    rows = []
    try:
        # Prewarm all workload shapes once; a new random prompt each time prevents
        # a benchmark round from reusing the previous round's prefix cache.
        for case in CASES:
            workload(llm, SamplingParams, args.engine, case, -1)
        for i in range(args.repeats):
            for case in CASES:
                row = workload(llm, SamplingParams, args.engine, case, i)
                rows.append(row)
                print(json.dumps({k: v for k, v in row.items() if k != "request_token_offsets_s"}), flush=True)
        data = {'engine': args.engine, 'package_path': package.__file__, 'model': args.model,
                'config': options, 'cases': CASES, 'sampling': {'temperature': .6, 'ignore_eos': True},
                'torch': torch.__version__, 'gpu': torch.cuda.get_device_name(),
                'target_kv_blocks': llm.model_runner.config.num_kvcache_blocks,
                'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
                'results': rows}
        with open(args.output, 'w') as f:
            json.dump(data, f, indent=2)
    finally:
        atexit.unregister(llm.exit)
        llm.exit()


if __name__ == '__main__':
    main()
