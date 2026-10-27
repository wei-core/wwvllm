# wwvllm

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
