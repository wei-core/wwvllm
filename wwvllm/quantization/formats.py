"""Validated 4-bit AWQ GEMM / GPTQ checkpoint-layout adapters."""
import torch
from wwvllm.quantization.linear import QuantizedLinear


def _unpack_last(packed, awq=False):
    shifts = torch.arange(8, dtype=torch.int64, device=packed.device) * 4
    values = (packed.to(torch.int64).unsqueeze(-1) >> shifts) & 15
    if awq:
        values = values[..., [0, 4, 1, 5, 2, 6, 3, 7]]
    return values.reshape(*packed.shape[:-1], -1)


def adapt_packed(qweight, qzeros, scales, method, group_size, g_idx=None):
    """Convert checkpoint tensors to logical (N,K), (N,G), (N,G), (K,)."""
    if method == 'awq':
        q = _unpack_last(qweight, awq=True)
        z = _unpack_last(qzeros, awq=True)
    elif method == 'gptq':
        # GPTQ packs input channels in qweight and stores zero points minus one.
        q = _unpack_last(qweight.T).T
        z = _unpack_last(qzeros) + 1
    else:
        raise ValueError('only AWQ GEMM and GPTQ 4-bit formats are supported')
    k, n = q.shape
    if scales.ndim != 2 or scales.shape[1] != n or z.shape != scales.shape:
        raise ValueError('quantized tensor dimensions do not match scales')
    groups = scales.shape[0]
    if g_idx is None:
        g_idx = torch.arange(k, dtype=torch.int32) // group_size
    g_idx = g_idx.to(dtype=torch.int32, device='cpu')
    if g_idx.numel() != k or int(g_idx.min()) < 0 or int(g_idx.max()) >= groups:
        raise ValueError('invalid group index')
    return q.T.contiguous().to(torch.uint8), scales.T.contiguous(), z.T.contiguous(), g_idx


def make_linear(parts, in_features, out_features, group_size, device, bias=None):
    qs, scales, zeros, group_indices = zip(*parts)
    if any(not torch.equal(group_indices[0], g) for g in group_indices[1:]):
        raise ValueError('merged projections must use the same g_idx')
    q = torch.cat(qs, 0)
    scales = torch.cat(scales, 0)
    zeros = torch.cat(zeros, 0)
    if q.shape != (out_features, in_features):
        raise ValueError(f'quantized matrix shape {tuple(q.shape)} != {(out_features, in_features)}')
    if in_features % 2:
        q = torch.nn.functional.pad(q, (0, 1))
    packed = q[:, ::2] | (q[:, 1::2] << 4)
    return QuantizedLinear(packed.to(device), scales.to(device=device, dtype=torch.float32),
                           in_features, 4, group_size, bias,
                           zeros.to(device=device, dtype=torch.float32), group_indices[0].to(device))
