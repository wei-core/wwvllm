import json
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file
from wwvllm.layers.moe import SparseMoE
from wwvllm.quantization.linear import QuantizedLinear
from wwvllm.utils.loader import SafeTensorsReader


def test_reader_index_duplicate_and_missing(tmp_path):
    save_file({'a': torch.ones(2)}, str(tmp_path / 'one.safetensors'))
    (tmp_path / 'model.safetensors.index.json').write_text(json.dumps({'weight_map': {'a': 'one.safetensors'}}))
    assert [k for k, _ in SafeTensorsReader(tmp_path).iter_tensors()] == ['a']
    (tmp_path / 'model.safetensors.index.json').write_text(json.dumps({'weight_map': {'a': 'missing.safetensors'}}))
    with pytest.raises(FileNotFoundError):
        SafeTensorsReader(tmp_path)
    (tmp_path / 'model.safetensors.index.json').unlink()
    save_file({'a': torch.ones(2)}, str(tmp_path / 'two.safetensors'))
    with pytest.raises(ValueError, match='duplicate'):
        list(SafeTensorsReader(tmp_path).iter_tensors())


@pytest.mark.parametrize('bits', [4, 8])
def test_quantized_linear_cpu_matches_dequant(bits):
    torch.manual_seed(1)
    linear = torch.nn.Linear(37, 19)
    q = QuantizedLinear.from_float(linear, bits, 16)
    x = torch.randn(3, 37)
    torch.testing.assert_close(q(x), torch.nn.functional.linear(x, q.dequantize(x.dtype), q.bias))
    assert q.qweight.numel() == (19 * (19 if bits == 4 else 37))


@pytest.mark.parametrize('normalize', [True, False])
def test_moe_router_and_dispatch_against_per_token(normalize):
    torch.manual_seed(12)
    moe = SparseMoE(8, 11, 4, 2, normalize)
    x = torch.randn(13, 8)
    ids, weights = moe.route(x)
    assert ids.shape == (13, 2)
    assert torch.all(ids[:, 0] != ids[:, 1])
    if normalize:
        torch.testing.assert_close(weights.sum(-1), torch.ones(13))
    y = moe(x)
    expected = torch.stack([sum(weights[t, j] * moe.experts[ids[t, j]](x[t]) for j in range(2)) for t in range(len(x))])
    torch.testing.assert_close(y, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')
@pytest.mark.parametrize('bits', [4, 8])
def test_quantized_triton_matches_dequant(bits):
    torch.manual_seed(2)
    q = QuantizedLinear.from_float(torch.nn.Linear(37, 51, device='cuda', dtype=torch.float16), bits, 16)
    for m in (1, 17, 65):
        x = torch.randn(m, 37, dtype=torch.float16, device='cuda')
        torch.testing.assert_close(q(x), torch.nn.functional.linear(x, q.dequantize(x.dtype), q.bias), atol=.003, rtol=.003)


def _pack(values):
    values = values.to(torch.int64)
    chunks = values.reshape(*values.shape[:-1], -1, 8)
    return ((chunks << (torch.arange(8, dtype=torch.int64) * 4)).sum(-1) & 0xffffffff).to(torch.int32)


@pytest.mark.parametrize('method', ['awq', 'gptq'])
def test_checkpoint_unpack_and_asymmetric_gemm_reference(method):
    from wwvllm.quantization.formats import adapt_packed, make_linear
    torch.manual_seed(6)
    k, n, group = 16, 8, 8
    q = torch.randint(0, 16, (k, n))
    zeros = torch.randint(1, 15, (2, n))
    scales = torch.rand(2, n) + .01
    if method == 'awq':
        inverse = [0, 2, 4, 6, 1, 3, 5, 7]
        qw = _pack(q.reshape(k, -1, 8)[..., inverse].reshape(k, -1))
        qz = _pack(zeros.reshape(2, -1, 8)[..., inverse].reshape(2, -1))
    else:
        qw = _pack(q.T).T.contiguous()
        qz = _pack(zeros - 1)
    g_idx = torch.tensor([1] * 8 + [0] * 8, dtype=torch.int32)
    parts = adapt_packed(qw, qz, scales, method, group, g_idx)
    layer = make_linear([parts], k, n, group, 'cpu')
    expected_weight = ((q - zeros[g_idx.long()]) * scales[g_idx.long()]).T
    torch.testing.assert_close(layer.dequantize(torch.float32), expected_weight)
    x = torch.randn(3, k)
    torch.testing.assert_close(layer(x), torch.nn.functional.linear(x, expected_weight))


def test_registry_explicit_compatibility():
    from wwvllm.models.registry import get_model_class
    for name in ('llama', 'qwen3', 'qwen3_moe'):
        assert get_model_class(SimpleNamespace(model_type=name, architectures=None))
    with pytest.raises(ValueError, match='unsupported model'):
        get_model_class(SimpleNamespace(model_type='unknown'))
    with pytest.raises(ValueError, match='scaled RoPE'):
        get_model_class(SimpleNamespace(model_type='llama', architectures=None, rope_scaling={'rope_type': 'linear'}))
    with pytest.raises(ValueError, match='sliding'):
        get_model_class(SimpleNamespace(model_type='qwen3', architectures=None, sliding_window=128))


def test_strict_loader_missing_and_shape_mismatch(tmp_path):
    from wwvllm.utils.loader import load_model
    model = torch.nn.Sequential(torch.nn.Linear(3, 2))
    path = tmp_path / 'model.safetensors'
    save_file({'0.weight': torch.ones(2, 3)}, str(path))
    with pytest.raises(ValueError, match='missing checkpoint parameters'):
        load_model(model, tmp_path)
    save_file({'0.weight': torch.ones(1, 3), '0.bias': torch.ones(2)}, str(path))
    with pytest.raises(ValueError, match='shape mismatch'):
        load_model(model, tmp_path)
    save_file({'0.weight': torch.ones(2, 3), '0.bias': torch.ones(2)}, str(path))
    load_model(model, tmp_path)
    assert torch.equal(model[0].weight, torch.ones(2, 3))


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')
@pytest.mark.parametrize('bits', [4, 8])
@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
def test_quantized_large_batch_arbitrary_groups(bits, dtype):
    torch.manual_seed(42)
    q = QuantizedLinear.from_float(torch.nn.Linear(37, 51, device='cuda', dtype=dtype), bits, 8)
    q.g_idx.copy_(torch.randint(0, q.scales.shape[1], (37,), device='cuda', dtype=torch.int32))
    q.zeros.copy_(torch.randint(0, 12, q.zeros.shape, device='cuda').to(q.zeros.dtype))
    for m in (4, 63, 64, 129):
        x = torch.randn(m, 37, device='cuda', dtype=dtype)
        torch.testing.assert_close(q(x), torch.nn.functional.linear(x, q.dequantize(dtype), q.bias),
                                   atol=.03, rtol=.03)
