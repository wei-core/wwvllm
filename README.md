# wwvllm

A lightweight inference engine with paged KV cache, prefix reuse, chunked
prefill, mixed prefill/decode scheduling, and optional speculative decoding.
The model registry supports Qwen3, Llama and Qwen3-MoE subject to their supported
attention and RoPE configurations. Quantized linear layers support groupwise
INT8/INT4 and selected AWQ/GPTQ checkpoint layouts.

## Installation

Use Linux with a CUDA-enabled PyTorch installation, Triton, a local model and
FlashAttention with paged KV support. Install FlashAttention separately for the
active PyTorch/CUDA environment, then install this package:

```bash
python -m pip install -e .
```

```python
from wwvllm import LLM, SamplingParams

llm = LLM('/path/to/model', prefill_chunk_size=512)
try:
    print(llm.generate(['Hello'], SamplingParams(temperature=0, max_tokens=32)))
finally:
    llm.exit()
```

For streaming, create the asynchronous engine inside an event loop:

```python
from wwvllm import AsyncLLMEngine, SamplingParams

async with AsyncLLMEngine('/path/to/model') as engine:
    request = engine.submit('Hello', SamplingParams(temperature=0, max_tokens=32))
    async for token_id in request.stream():
        print(token_id)
    result = await request.result()
```

The unified token budget covers both prefill and decode. Quantized execution,
the MoE reference backend and speculative decoding currently require one GPU.
Speculative decoding requires matching Qwen3 target and draft vocabularies;
its draft cache reconstruction can add substantial overhead. Quantization after
loading a floating-point checkpoint requires memory for the original weights.
The optional mixed-batch CUDA Graph cache is disabled by default.

The `engine`, `core`, `executor`, `models`, `layers`, and `quantization`
packages separate request lifecycle, scheduling, execution and model code.
Benchmark scripts are in `benchmarks/`; tests and detailed engineering records
are maintained separately.
