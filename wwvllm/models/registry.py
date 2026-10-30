"""Select an explicitly supported model; reject unimplemented attention variants."""
from wwvllm.models.qwen3 import Qwen3ForCausalLM
from wwvllm.models.llama import LlamaForCausalLM
from wwvllm.models.qwen3_moe import Qwen3MoeForCausalLM

_MODELS = {'qwen3': ('Qwen3ForCausalLM', Qwen3ForCausalLM),
           'llama': ('LlamaForCausalLM', LlamaForCausalLM),
           'qwen3_moe': ('Qwen3MoeForCausalLM', Qwen3MoeForCausalLM)}


def get_model_class(config):
    model_type = getattr(config, 'model_type', None)
    if model_type not in _MODELS:
        raise ValueError(f'unsupported model type: {model_type}')
    name, cls = _MODELS[model_type]
    architectures = getattr(config, 'architectures', None)
    if architectures and name not in architectures:
        raise ValueError(f'unsupported architecture for {model_type}: {architectures}')
    rope = getattr(config, 'rope_scaling', None)
    if rope and rope.get('rope_type', rope.get('type', 'default')) != 'default':
        raise ValueError('scaled RoPE is not supported')
    if getattr(config, 'use_sliding_window', False) or getattr(config, 'sliding_window', None):
        raise ValueError('sliding-window attention is not supported')
    if getattr(config, 'hidden_act', 'silu') != 'silu':
        raise ValueError('only SiLU SwiGLU is supported')
    return cls
