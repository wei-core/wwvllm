"""Online benchmark: python -m wwvllm.benchmarks.online MODEL [options]."""
import argparse
import asyncio
import json
from time import monotonic

from wwvllm import AsyncLLMEngine, SamplingParams


def percentile(values, p):
    if not values:
        return None
    values = sorted(values)
    return values[round((len(values) - 1) * p)]


async def benchmark(args):
    async with AsyncLLMEngine(args.model, enforce_eager=args.eager,
                              prefill_chunk_size=args.chunk, max_model_len=args.max_model_len) as engine:
        started = monotonic()
        handles = [engine.submit([1] * args.input_len, SamplingParams(temperature=0, max_tokens=args.output_len, ignore_eos=True)) for _ in range(args.requests)]
        results = await asyncio.gather(*(h.result() for h in handles))
        elapsed = monotonic() - started
    ttft = [r['token_timestamps'][0] - r['arrival_time'] for r in results]
    tpot = [(r['token_timestamps'][-1] - r['token_timestamps'][0]) / (len(r['token_timestamps']) - 1) for r in results if len(r['token_timestamps']) > 1]
    itl = [b - a for r in results for a, b in zip(r['token_timestamps'], r['token_timestamps'][1:])]
    print(json.dumps({'requests': args.requests, 'elapsed_s': elapsed, 'output_tokens_per_s': sum(len(r['token_ids']) for r in results) / elapsed,
                      **{f'{name}_p{int(p*100)}_s': percentile(values, p) for name, values in [('ttft', ttft), ('tpot', tpot), ('itl', itl)] for p in (.5, .95, .99)},
                      'preemptions': sum(r['preemption_count'] for r in results)}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model')
    parser.add_argument('--requests', type=int, default=4)
    parser.add_argument('--input-len', type=int, default=128)
    parser.add_argument('--output-len', type=int, default=32)
    parser.add_argument('--chunk', type=int, default=512)
    parser.add_argument('--max-model-len', type=int, default=4096)
    parser.add_argument('--eager', action='store_true')
    asyncio.run(benchmark(parser.parse_args()))
