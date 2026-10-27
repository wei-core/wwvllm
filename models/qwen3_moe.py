"""Qwen3 MoE adapter with reference PyTorch expert dispatch."""
import torch
from torch import nn
import torch.distributed as dist
from wwvllm.layers.moe import SparseMoE
from wwvllm.layers.embed_head import VocabParallelEmbedding, ParallelLMHead
from wwvllm.layers.layernorm import RMSNorm
from wwvllm.models.qwen3 import Qwen3DecoderLayer, Qwen3ForCausalLM


class Qwen3MoeModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        if dist.get_world_size() != 1:
            raise ValueError('Qwen3 MoE reference backend requires tensor_parallel_size=1')
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList()
        for i in range(config.num_hidden_layers):
            layer = Qwen3DecoderLayer(config)
            if (i not in config.mlp_only_layers and config.num_experts > 0
                    and (i + 1) % config.decoder_sparse_step == 0):
                layer.mlp = SparseMoE(config.hidden_size, config.moe_intermediate_size,
                                      config.num_experts, config.num_experts_per_tok,
                                      config.norm_topk_prob)
            self.layers.append(layer)
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, input_ids, positions):
        hidden = self.embed_tokens(input_ids)
        residual = None
        for layer in self.layers:
            hidden, residual = layer(positions, hidden, residual)
        hidden, _ = self.norm(hidden, residual)
        return hidden


class Qwen3MoeForCausalLM(Qwen3ForCausalLM):
    def __init__(self, config):
        nn.Module.__init__(self)
        self.model = Qwen3MoeModel(config)
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
        if config.tie_word_embeddings:
            self.lm_head.weight.data = self.model.embed_tokens.weight.data
