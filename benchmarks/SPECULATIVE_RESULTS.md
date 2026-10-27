# Speculative decoding smoke comparison

2026-10-27; RTX 5060 (shared GPU), Qwen3-0.6B as **both** target and draft, greedy, K=2, input 32 tokens, output 16 tokens, batch size 1, eager, one warmed run for each mode. Both models were loaded in both measurements. This is a functional comparison, not evidence for production throughput or a smaller-draft speedup.

```sh
PYTHONPATH=.. python -m wwvllm.benchmarks.speculative \
  /home/ubuntu/ai-infra/minivllm/Models/Qwen3-0.6B \
  --draft /home/ubuntu/ai-infra/minivllm/Models/Qwen3-0.6B \
  --input-len 32 --output-len 16 --k 2
```

| Metric | Normal | Speculative |
| --- | ---: | ---: |
| Wall time | 0.993 s | 2.188 s |
| Output throughput | 16.11 token/s | 7.31 token/s |
| Scheduler steps | 16 | 6 |
| Draft accepted / proposed | — | 10 / 10 |
| Target verification forwards | — | 5 |

Outputs matched. Speculative throughput was 0.454x normal (about 54.6% lower), despite all proposals being accepted. Peak CUDA allocated memory during the combined run was 5,182,286,336 bytes; it is not a per-mode peak. Rebuilding the draft KV every verification and using a draft as large as the target are likely significant overheads, but this run does not attribute time by component. Repeated tests with a genuinely smaller, tokenizer-compatible draft, varying concurrency and K, and separately measured memory and latencies are needed before claiming a speedup.
