import os
from dataclasses import dataclass
from transformers import AutoConfig


@dataclass(slots=True)
class Config:
    model: str
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 512
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
    hf_config: AutoConfig | None = None
    eos: int = -1
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1

    def __post_init__(self):
        if min(self.max_num_batched_tokens, self.max_num_seqs, self.prefill_chunk_size, self.max_pending_requests) < 1:
            raise ValueError("scheduler limits must be positive")
        if self.enable_speculative:
            if not self.speculative_model or not os.path.isdir(self.speculative_model):
                raise ValueError("speculative_model must be an existing model directory")
            if self.tensor_parallel_size != 1:
                raise ValueError("speculative decoding currently requires tensor_parallel_size=1")
            if self.speculative_num_tokens < 1 or not 0 < self.draft_memory_reserve < self.gpu_memory_utilization:
                raise ValueError("invalid speculative_num_tokens or draft_memory_reserve")
        assert os.path.isdir(self.model)
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        self.hf_config = AutoConfig.from_pretrained(self.model)
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
