"""Single-request warm comparison; see --help for options."""
import argparse
import json
import time
import torch
from wwvllm import LLM, SamplingParams


def run(engine, prompt, tokens, enabled):
    params = SamplingParams(temperature=0, max_tokens=tokens, ignore_eos=True, speculative=enabled)
    torch.cuda.synchronize()
    start = time.perf_counter()
    seq = engine.add_request(prompt, params)
    steps = 0
    while not engine.is_finished():
        engine.step()
        steps += 1
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return {'seconds': elapsed, 'output_tokens_per_s': seq.num_completion_tokens / elapsed,
            'output_ids': seq.completion_token_ids, 'scheduler_steps': steps,
            'draft_proposed': seq.draft_proposed, 'draft_accepted': seq.draft_accepted,
            'target_verifications': seq.target_verifications}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model')
    parser.add_argument('--draft', required=True)
    parser.add_argument('--k', type=int, default=2)
    parser.add_argument('--input-len', type=int, default=32)
    parser.add_argument('--output-len', type=int, default=16)
    parser.add_argument('--gpu-memory-utilization', type=float, default=.85)
    args = parser.parse_args()
    engine = LLM(args.model, enable_speculative=True, speculative_model=args.draft,
                 speculative_num_tokens=args.k, enforce_eager=True,
                 max_num_batched_tokens=128, max_model_len=max(256, args.input_len + args.output_len),
                 max_num_seqs=1, gpu_memory_utilization=args.gpu_memory_utilization)
    prompt = [10 + i % 50 for i in range(args.input_len)]
    try:
        run(engine, prompt, 4, False)
        run(engine, prompt, 4, True)
        normal = run(engine, prompt, args.output_len, False)
        speculative = run(engine, prompt, args.output_len, True)
        assert normal['output_ids'] == speculative['output_ids'], 'greedy outputs differ'
        print(json.dumps({'normal': normal, 'speculative': speculative,
                          'speedup': speculative['output_tokens_per_s'] / normal['output_tokens_per_s'],
                          'peak_allocated_bytes': torch.cuda.max_memory_allocated()}, indent=2))
    finally:
        engine.exit()


if __name__ == '__main__':
    main()
