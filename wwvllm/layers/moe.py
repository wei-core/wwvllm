"""Single-device sparse MoE reference backend."""
import torch
from torch import nn
import torch.nn.functional as F


class ExpertMLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class TorchMoEBackend(nn.Module):
    def forward(self, x, experts, indices, weights):
        output = torch.zeros_like(x)
        for i, expert in enumerate(experts):
            token, slot = torch.where(indices == i)
            if token.numel():
                value = expert(x.index_select(0, token)) * weights[token, slot, None]
                output.index_add_(0, token, value.to(output.dtype))
        return output


class SparseMoE(nn.Module):
    def __init__(self, hidden_size, intermediate_size, num_experts, top_k, norm_topk_prob):
        super().__init__()
        if not 1 <= top_k <= num_experts:
            raise ValueError('invalid MoE top-k')
        self.gate = nn.Linear(hidden_size, num_experts, bias=False)
        self.experts = nn.ModuleList([ExpertMLP(hidden_size, intermediate_size) for _ in range(num_experts)])
        self.top_k = top_k
        self.norm_topk_prob = norm_topk_prob
        self.backend = TorchMoEBackend()

    def route(self, x):
        probs = F.softmax(self.gate(x), dim=-1, dtype=torch.float32)
        weights, indices = probs.topk(self.top_k, dim=-1)
        if self.norm_topk_prob:
            weights = weights / weights.sum(dim=-1, keepdim=True)
        return indices, weights.to(x.dtype)

    def forward(self, x):
        shape = x.shape
        x = x.reshape(-1, shape[-1])
        indices, weights = self.route(x)
        return self.backend(x, self.experts, indices, weights).reshape(shape)
