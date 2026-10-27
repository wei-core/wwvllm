# Stage 3 functional performance observations

2026-10-27, RTX 5060 8 GiB, PyTorch 2.9.1+cu128, Transformers 4.57.3, FlashAttention 2.8.3. GPU was shared. Each row is one separately initialized process with one warmed request, 32 input tokens, 16 output tokens, greedy, `enforce_eager=True`, and `max_model_len=256`. This is not a statistically reliable throughput comparison.

| Model / mode | Output tok/s | TTFT (s) | TPOT (s) | Model tensor bytes | Allocated KV bytes | Peak allocated bytes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3-0.6B BF16 | 17.77 | 0.059 | 0.056 | 1,213,071,360 | 3,875,536,896 | 5,098,667,008 |
| Qwen3-0.6B INT8 | 14.51 | 0.072 | 0.069 | 800,997,376 | 3,082,813,440 | 4,168,596,480 |
| Qwen3-0.6B INT4 | 13.76 | 0.107 | 0.070 | 580,796,416 | 3,170,893,824 | 4,042,243,072 |
| Huihui Qwen3-MoE BF16, first run | 3.42 | 0.364 | 0.288 | 2,270,208,000 | 2,583,691,264 | 4,901,706,752 |
| Same MoE, second run | 6.22 | 0.169 | 0.160 | 2,270,208,000 | 2,583,691,264 | 4,901,706,752 |

On this sample INT8 throughput was about 18.3% lower than BF16 and INT4 about 22.5% lower; model tensor storage fell about 34.0% and 52.1%, respectively. Model bytes include embeddings, norms, scale/zero-point buffers and rotary buffers, not only quantized weights. KV allocation depends on available GPU memory and warmup peaks, and is **not fixed across runs**; peak values therefore are not a controlled measurement of quantization memory savings. INT8 happened to produce the same 16 greedy tokens as BF16 in this prompt; INT4 produced different tokens. Neither result establishes quality or accuracy generally.

MoE and Dense are different checkpoints and sizes; their throughput is not an architecture-only comparison. A separate instrumented MoE request (32 input, 4 output, top-1, 28 layers) recorded aggregate expert assignments `[318, 321, 341]`; instrumentation is outside the timed section and adds synchronization. The two MoE results illustrate run-to-run variation on the shared GPU.

Reproduce each mode in a fresh process:

```bash
PYTHONPATH=.. python -m wwvllm.benchmarks.stage3 /path/to/Qwen3 --input-len 32 --output-len 16
PYTHONPATH=.. python -m wwvllm.benchmarks.stage3 /path/to/Qwen3 --quantization int8
PYTHONPATH=.. python -m wwvllm.benchmarks.stage3 /path/to/Qwen3 --quantization int4
PYTHONPATH=.. python -m wwvllm.benchmarks.stage3 /path/to/Qwen3-MoE --collect-experts
```

Measure repeated trials with a fixed GPU load, fixed KV allocation, representative prompts, both prefill and decode workloads, and quality metrics before drawing performance or quality conclusions.
