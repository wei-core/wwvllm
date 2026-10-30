"""Weight-only groupwise symmetric quantization with a CUDA Triton GEMM."""
import torch
from torch import nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _quant_gemm(X, Q, S, Z, GI, Y, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                GROUP: tl.constexpr, NG: tl.constexpr, PACKED: tl.constexpr,
                BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    mi = tl.program_id(0) * BM + tl.arange(0, BM)
    ni = tl.program_id(1) * BN + tl.arange(0, BN)
    kk = tl.arange(0, BK)
    acc = tl.full((BM, BN), 0, tl.float32)
    for block in range(tl.cdiv(K, BK)):
        ki = block * BK + kk
        x = tl.load(X + mi[:, None] * K + ki[None, :],
                    (mi[:, None] < M) & (ki[None, :] < K), 0)
        if PACKED:
            raw = tl.load(Q + ni[None, :] * tl.cdiv(K, 2) + ki[:, None] // 2,
                          (ni[None, :] < N) & (ki[:, None] < K), 0).to(tl.int32)
            q = (raw >> ((ki[:, None] % 2) * 4)) & 15
        else:
            q = tl.load(Q + ni[None, :] * K + ki[:, None],
                        (ni[None, :] < N) & (ki[:, None] < K), 0).to(tl.int32)
        group = tl.load(GI + ki, ki < K, 0)
        scale = tl.load(S + ni[None, :] * NG + group[:, None],
                        (ni[None, :] < N) & (ki[:, None] < K), 0).to(tl.float32)
        zero = tl.load(Z + ni[None, :] * NG + group[:, None],
                       (ni[None, :] < N) & (ki[:, None] < K), 0).to(tl.float32)
        w = ((q.to(tl.float32) - zero) * scale).to(x.dtype)
        acc += tl.dot(x, w)
    tl.store(Y + mi[:, None] * N + ni[None, :], acc,
             (mi[:, None] < M) & (ni[None, :] < N))


class QuantizedLinear(nn.Module):
    def __init__(self, qweight, scales, in_features, bits, group_size, bias=None, zeros=None, g_idx=None):
        super().__init__()
        self.register_buffer('qweight', qweight.contiguous())
        self.register_buffer('scales', scales.contiguous())
        self.register_buffer('bias', bias)
        self.register_buffer('zeros', zeros if zeros is not None else torch.full_like(scales, 8 if bits == 4 else 0))
        self.register_buffer('g_idx', g_idx if g_idx is not None else torch.arange(in_features, device=qweight.device, dtype=torch.int32) // group_size)
        self.in_features = in_features
        self.out_features = qweight.shape[0]
        self.bits = bits
        self.group_size = group_size

    @classmethod
    def from_float(cls, linear, bits=8, group_size=128):
        if bits not in (4, 8) or group_size < 1:
            raise ValueError('expected int4/int8 and a positive group size')
        weight = linear.weight.detach().float()
        n, k = weight.shape
        groups = (k + group_size - 1) // group_size
        padded = F.pad(weight, (0, groups * group_size - k)).reshape(n, groups, group_size)
        limit = 7 if bits == 4 else 127
        scale = padded.abs().amax(-1).clamp_min(1e-8) / limit
        q = (padded / scale.unsqueeze(-1)).round().clamp(-limit, limit).to(torch.int8).reshape(n, -1)[:, :k].contiguous()
        if bits == 4:
            if k % 2:
                q = F.pad(q, (0, 1))
            q = (q.to(torch.int16) + 8).to(torch.uint8)
            q = q[:, 0::2] | (q[:, 1::2] << 4)
        bias = linear.bias.detach().clone() if linear.bias is not None else None
        return cls(q, scale, k, bits, group_size, bias)

    def dequantize(self, dtype):
        if self.bits == 4:
            q = torch.stack((self.qweight & 15, self.qweight >> 4), -1).reshape(self.out_features, -1)[:, :self.in_features].to(torch.int16)
        else:
            q = self.qweight
        groups = self.g_idx.long()
        return ((q.float() - self.zeros[:, groups]) * self.scales[:, groups]).to(dtype)

    def forward(self, x):
        shape = x.shape
        x = x.reshape(-1, self.in_features).contiguous()
        if x.is_cuda and x.dtype in (torch.float16, torch.bfloat16):
            out = torch.empty((x.shape[0], self.out_features), dtype=x.dtype, device=x.device)
            # Larger prefill batches benefit from reusing each weight tile across
            # more rows; retain the original tile for small decode batches.
            block_m = 32 if x.shape[0] >= 64 else 16
            _quant_gemm[(triton.cdiv(x.shape[0], block_m), triton.cdiv(self.out_features, 32))](
                x, self.qweight, self.scales, self.zeros, self.g_idx, out, x.shape[0], self.out_features, self.in_features,
                self.group_size, self.scales.shape[1], self.bits == 4, block_m, 32, 32)
            if self.bias is not None:
                out = out + self.bias
        else:
            out = F.linear(x, self.dequantize(x.dtype), self.bias)
        return out.reshape(*shape[:-1], self.out_features)


def quantize_model(model, bits, group_size):
    from wwvllm.layers.linear import LinearBase
    for name, module in list(model.named_children()):
        if isinstance(module, LinearBase):
            if module.tp_size != 1:
                raise ValueError('weight-only quantization requires tensor_parallel_size=1')
            setattr(model, name, QuantizedLinear.from_float(module, bits, group_size))
        else:
            quantize_model(module, bits, group_size)
