# wwvllm

魔改vllm。

Qwen3 inference with paged KV cache, prefix reuse, chunked prefill and mixed prefill/decode scheduling. All imports refer to this repository's `wwvllm` package; run commands from its parent or add that parent to `PYTHONPATH`. Requires PyTorch, Transformers, Triton, FlashAttention with paged KV support, xxhash and NumPy; CUDA and a local Qwen3 model are required for inference.

```python
from wwvllm import LLM, SamplingParams

llm = LLM('/path/to/Qwen3', prefill_chunk_size=512)
try:
    print(llm.generate(['Hello'], SamplingParams(temperature=0, max_tokens=32)))
finally:
    llm.exit()
```

For online use, create the engine inside an asyncio event loop. One worker owns the model and executes one batch at a time. `submit` returns immediately; the handle supports a final result, token stream and cancellation. The stream yields token IDs; the result contains decoded text, IDs and timestamps. Close the engine before closing the event loop.

```python
from wwvllm import AsyncLLMEngine, SamplingParams

async with AsyncLLMEngine('/path/to/Qwen3', max_pending_requests=1024) as engine:
    request = engine.submit('Hello', SamplingParams(temperature=0, max_tokens=32))
    async for token_id in request.stream():
        print(token_id)
    result = await request.result()
```

The unified `max_num_batched_tokens` budget includes decode and prefill. Decode is prioritized; each prefill chunk is limited by `prefill_chunk_size`. Requests exceeding `max_model_len` or the available KV cache are rejected. Mixed batches use eager varlen paged attention; pure decode retains CUDA graphs unless `enforce_eager=True`. Greedy sampling uses `temperature=0`. Tensor parallel workers receive full token histories to allow recomputation after preemption.

Run CPU scheduling and lifecycle tests with `PYTHONPATH=.. pytest -q tests`. For a repeatable online measurement, run `PYTHONPATH=.. python -m wwvllm.benchmarks.online /path/to/Qwen3 --eager`; it reports output throughput, TTFT, TPOT and inter-token latency percentiles in seconds. Adjust input/output lengths and concurrent requests to suit the model and GPU.

## Speculative decoding (single GPU)

Enable the optional draft model with `enable_speculative=True, speculative_model='/path/to/draft', speculative_num_tokens=2`. Both models must be Qwen3 with identical tokenizer vocabularies and special tokens. This initial implementation supports tensor parallel size 1. `SamplingParams(speculative=False)` selects normal decoding for an individual request; temperature, `top_k` (0 disables), and `top_p` apply to both models. The draft model reconstructs its independent KV cache from committed tokens each round and discards it after proposing, trading simplicity for substantial overhead. `draft_memory_reserve` reserves a fraction of device memory beyond the target KV budget, but does not guarantee enough space for all workspaces or long contexts.

```python
llm = LLM('/path/to/target', enable_speculative=True,
          speculative_model='/path/to/draft', speculative_num_tokens=2)
try:
    print(llm.generate(['Hello'], SamplingParams(temperature=0, max_tokens=32)))
finally:
    llm.exit()
```

Run `PYTHONPATH=.. python -m wwvllm.benchmarks.speculative TARGET --draft DRAFT` for a warmed, single-request comparison. Draft and target load simultaneously in both modes, so this measures decoding rather than the full memory/initialization cost of enabling speculation. It does not replace multi-load or repeated statistical benchmarking.

## Stage 3 models and quantization

The model registry supports Qwen3, Llama and Qwen3-MoE with standard full attention, SiLU and unscaled RoPE. Unsupported attention/RoPE configurations are rejected. Qwen3-MoE uses a single-GPU PyTorch expert backend and automatically runs eagerly. Dense floating-point tensor parallelism retains its original path.

`LLM(path, quantization='int8')` and `quantization='int4'` perform groupwise symmetric weight-only quantization of the engine's parallel linear layers **after loading a floating-point checkpoint**. Embeddings, LM head and MoE expert/router `nn.Linear` layers stay floating point. The CUDA path uses a Triton W8A16/W4A16 GEMM. This approach needs enough temporary memory for the floating model during load; the final weight savings do not describe peak initialization memory. `quant_group_size` defaults to 128. Quantization currently requires TP=1.

A checkpoint with `quantization_config` declaring 4-bit AWQ GEMM with explicit zero points or legacy GPTQ is detected automatically; it is loaded into the same groupwise runtime layout. These adapters currently cover the explicit layouts in `quantization/formats.py`, including GPTQ `g_idx`; unsupported variants and quantized MoE experts are rejected. They have been tested with generated checkpoints, not a third-party production AWQ/GPTQ checkpoint. All checkpoint paths require safetensors and strict parameter coverage.

See [stage 3 notes](docs/阶段三改造说明.md) and [measured results](benchmarks/STAGE3_RESULTS.md) for support boundaries, numerical tests, and observed slowdowns.

## Optional mixed-batch CUDA Graph cache

`enable_mixed_graph=True` enables bounded, lazy CUDA Graph reuse for eligible Qwen3 mixed prefill/decode batches on a single GPU with floating-point weights and CUDA Graph enabled. `max_mixed_graphs` defaults to 16. The default remains off: unfamiliar shapes incur capture cost and graph allocations, and cache eviction can cause recapture. Pure prefill, pure decode, tensor-parallel, quantized, other-architecture and eager paths retain their existing execution behavior. See [the optimization report](docs/性能调优/阶段一/03-混合图复用/README.md) for measured warm and cold results, limitations and reproduction.
