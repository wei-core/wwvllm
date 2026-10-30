import os
from dataclasses import dataclass
from transformers import AutoConfig


@dataclass(slots=True)
class Config:
    model: str
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 512
    quantization: str | None = None
    quant_group_size: int = 128
    prefill_chunk_size: int = 512
    max_pending_requests: int = 1024
    enable_speculative: bool = False
    speculative_model: str | None = None
    speculative_num_tokens: int = 4
    draft_memory_reserve: float = 0.10
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    enforce_eager: bool = False
    enable_mixed_graph: bool = False
    max_mixed_graphs: int = 16
    hf_config: AutoConfig | None = None
    eos: int = -1
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1

    def __post_init__(self):
        if min(self.max_num_batched_tokens, self.max_num_seqs, self.prefill_chunk_size, self.max_pending_requests) < 1:
            raise ValueError("scheduler limits must be positive")
        if self.max_mixed_graphs < 1:
            raise ValueError("max_mixed_graphs must be positive")
        if self.quantization not in (None, 'int8', 'int4', 'awq', 'gptq') or self.quant_group_size < 1:
            raise ValueError("quantization must be None, int8, int4, awq or gptq; group size must be positive")
        if self.quantization and self.tensor_parallel_size != 1:
            raise ValueError("quantized execution requires tensor_parallel_size=1")
        if self.enable_speculative:
            if not self.speculative_model or not os.path.isdir(self.speculative_model):
                raise ValueError("speculative_model must be an existing model directory")
            if self.tensor_parallel_size != 1:
                raise ValueError("speculative decoding currently requires tensor_parallel_size=1")
            if self.speculative_num_tokens < 1 or not 0 < self.draft_memory_reserve < self.gpu_memory_utilization:
                raise ValueError("invalid speculative_num_tokens or draft_memory_reserve")
        if not os.path.isdir(self.model):
            raise ValueError("model must be an existing directory")
        if self.kvcache_block_size < 256 or self.kvcache_block_size % 256:
            raise ValueError("kvcache_block_size must be a positive multiple of 256")
        if not 1 <= self.tensor_parallel_size <= 8:
            raise ValueError("tensor_parallel_size must be between 1 and 8")
        self.hf_config = AutoConfig.from_pretrained(self.model)
        if getattr(self.hf_config, 'model_type', None) == 'qwen3_moe' and self.tensor_parallel_size != 1:
            raise ValueError("Qwen3 MoE reference backend requires tensor_parallel_size=1")
        metadata = getattr(self.hf_config, 'quantization_config', None)
        if metadata:
            method = metadata.get('quant_method', '').lower()
            if method not in ('awq', 'gptq') or self.quantization not in (None, method):
                raise ValueError("unsupported or mismatched quantized checkpoint format")
            if self.tensor_parallel_size != 1:
                raise ValueError("quantized execution requires tensor_parallel_size=1")
            self.quantization = method
        elif self.quantization in ('awq', 'gptq'):
            raise ValueError("AWQ/GPTQ requires checkpoint quantization metadata")
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
