# Smoke benchmark

2026-10-27, one RTX 5060, Qwen3-0.6B, PyTorch 2.9.1+cu128, FlashAttention 2.8.3. The GPU was shared with other workloads. This is a single cold-start functional measurement, not a performance comparison; compilation and warmup affect it.

```sh
PYTHONPATH=.. python -m wwvllm.benchmarks.online /home/ubuntu/ai-infra/minivllm/Models/Qwen3-0.6B --eager --requests 2 --input-len 32 --output-len 5 --chunk 8 --max-model-len 256
```

Output throughput: 2.12 tokens/s; TTFT P50/P95: 4.445/4.445 s; TPOT P50/P95: 0.068/0.068 s; ITL P50/P95: 0.064/0.100 s; preemptions: 0. Measurements are end-to-end request timings, including the first invocation's compilation. Run repeated warmed trials and compare a fixed baseline before drawing conclusions about performance.
