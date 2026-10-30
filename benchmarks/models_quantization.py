"""One warmed single-request run. Run each model/mode in a fresh process."""
import argparse
import json
import time
import torch
from wwvllm import LLM, SamplingParams


def bytes_used(tensors):
    seen = set()
    total = 0
    for t in tensors:
        storage = t.untyped_storage()
        key = (storage.data_ptr(), storage.nbytes())
        if key not in seen:
            seen.add(key)
            total += storage.nbytes()
    return total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model')
    parser.add_argument('--quantization', choices=['int4', 'int8'])
    parser.add_argument('--input-len', type=int, default=32)
    parser.add_argument('--output-len', type=int, default=16)
    parser.add_argument('--gpu-memory-utilization', type=float, default=.75)
    parser.add_argument('--collect-experts', action='store_true')
    args = parser.parse_args()
    engine = LLM(args.model, quantization=args.quantization, enforce_eager=True,
                 max_num_batched_tokens=128, max_model_len=256, max_num_seqs=2,
                 gpu_memory_utilization=args.gpu_memory_utilization)
    params = SamplingParams(temperature=0, max_tokens=args.output_len, ignore_eos=True)
    prompt = [10 + i % 50 for i in range(args.input_len)]
    try:
        model = engine.model_runner.model
        weights = bytes_used(list(model.parameters()) + list(model.buffers()))
        kv = engine.model_runner.kv_cache.numel() * engine.model_runner.kv_cache.element_size()
        engine.generate([prompt], SamplingParams(temperature=0, max_tokens=4, ignore_eos=True), use_tqdm=False)
        torch.cuda.synchronize()
        started = time.perf_counter()
        seq = engine.add_request(prompt, params)
        while not engine.is_finished():
            engine.step()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        times = seq.token_timestamps
        result = {'model': args.model, 'quantization': args.quantization, 'input_tokens': args.input_len,
                  'output_tokens': len(seq.completion_token_ids), 'seconds': elapsed,
                  'output_tokens_per_s': len(seq.completion_token_ids) / elapsed,
                  'ttft_s': times[0] - seq.arrival_time,
                  'tpot_s': (times[-1] - times[0]) / (len(times) - 1) if len(times) > 1 else None,
                  'model_tensor_bytes': weights, 'target_kv_bytes': kv,
                  'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated(),
                  'token_ids': seq.completion_token_ids}
        if args.collect_experts:
            from wwvllm.layers.moe import SparseMoE
            counts = {}
            for name, layer in model.named_modules():
                if isinstance(layer, SparseMoE):
                    original = layer.route
                    def route(x, original=original, name=name, size=layer.gate.out_features):
                        indices, weights = original(x)
                        current = torch.bincount(indices.reshape(-1), minlength=size).cpu().tolist()
                        counts[name] = [a + b for a, b in zip(counts.get(name, [0] * size), current)]
                        return indices, weights
                    layer.route = route
            engine.generate([prompt], SamplingParams(temperature=0, max_tokens=4, ignore_eos=True), use_tqdm=False)
            result['expert_counts_separate_run'] = counts
            result['expert_totals_separate_run'] = [sum(v[i] for v in counts.values()) for i in range(len(next(iter(counts.values()))))] if counts else []
        print(json.dumps(result, indent=2))
    finally:
        engine.exit()


if __name__ == '__main__':
    main()
