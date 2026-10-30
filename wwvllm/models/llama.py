"""Llama decoder using the shared paged-attention and SwiGLU primitives."""
from wwvllm.models.qwen3 import Qwen3ForCausalLM


class LlamaForCausalLM(Qwen3ForCausalLM):
    pass
