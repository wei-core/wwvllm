"""Load the supported AWQ GEMM / GPTQ 4-bit variants into runtime layout."""
from collections import defaultdict
from wwvllm.models.loader import SafeTensorsReader, default_weight_loader
from wwvllm.quantization.formats import adapt_packed, make_linear
from wwvllm.layers.linear import LinearBase


def load_quantized_model(model, path, metadata):
    method = metadata.get('quant_method', '').lower()
    bits = metadata.get('bits', 4)
    group_size = metadata.get('group_size', 128)
    if method not in ('awq', 'gptq') or bits != 4 or group_size <= 0:
        raise ValueError('only groupwise 4-bit AWQ/GPTQ checkpoints are supported')
    if method == 'gptq' and str(metadata.get('checkpoint_format', 'gptq')).lower() != 'gptq':
        raise ValueError('only the legacy GPTQ zero-point encoding is supported')
    if method == 'awq' and (str(metadata.get('version', 'gemm')).lower() != 'gemm' or not metadata.get('zero_point', True)):
        raise ValueError('only AWQ GEMM with explicit zero points is supported')
    params = dict(model.named_parameters(remove_duplicate=False))
    mapping = getattr(model, 'packed_modules_mapping', {})
    loaded = defaultdict(set)
    bundles = defaultdict(dict)
    for name, tensor in SafeTensorsReader(path).iter_tensors():
        base, _, suffix = name.rpartition('.')
        if suffix in ('qweight', 'qzeros', 'scales', 'g_idx'):
            if '.experts.' in name:
                raise ValueError('pre-quantized MoE experts are not supported')
            bundles[base][suffix] = tensor
            continue
        target, shard = name, None
        components = name.split('.')
        if '.experts.' not in name:
            for i, part in enumerate(components):
                if part in mapping:
                    components[i], shard = mapping[part]
                    target = '.'.join(components)
                    break
        if target not in params or shard in loaded[target]:
            raise ValueError(f'unexpected or duplicate tensor: {name}')
        param = params[target]
        loader = getattr(param, 'weight_loader', default_weight_loader)
        loader(param, tensor, shard) if shard is not None else loader(param, tensor)
        loaded[target].add(shard)
    merged = defaultdict(dict)
    for base, values in bundles.items():
        if not {'qweight', 'qzeros', 'scales'} <= values.keys():
            raise ValueError(f'incomplete quantized tensors: {base}')
        components = base.split('.')
        shard = None
        for i, part in enumerate(components):
            if part in mapping:
                components[i], shard = mapping[part]
                break
        target = '.'.join(components)
        if shard in merged[target]:
            raise ValueError(f'duplicate quantized projection: {base}')
        merged[target][shard] = adapt_packed(values['qweight'], values['qzeros'], values['scales'],
                                              method, group_size, values.get('g_idx'))
    for name, parts in merged.items():
        if loaded[name + '.weight']:
            raise ValueError(f'duplicate floating and quantized weight: {name}')
        old = model.get_submodule(name)
        if not isinstance(old, LinearBase) or old.tp_size != 1:
            raise ValueError(f'unsupported quantized projection: {name}')
        if 'qkv_proj' == name.split('.')[-1]:
            order = ['q', 'k', 'v']
        elif 'gate_up_proj' == name.split('.')[-1]:
            order = [0, 1]
        else:
            order = [None]
        if set(parts) != set(order):
            raise ValueError(f'missing quantized projection shards: {name}')
        if old.bias is not None and name + '.bias' not in loaded:
            raise ValueError(f'missing bias for {name}')
        bias = old.bias.detach().clone() if old.bias is not None else None
        replacement = make_linear([parts[i] for i in order], old.weight.shape[1], old.weight.shape[0],
                                  group_size, old.weight.device, bias)
        parent_name, _, leaf = name.rpartition('.')
        setattr(model.get_submodule(parent_name), leaf, replacement)
        loaded[name + '.weight'] = {None}
    missing = []
    for name, param in params.items():
        if not loaded[name]:
            if name == 'lm_head.weight' and loaded['model.embed_tokens.weight'] and \
                    param.data_ptr() == params['model.embed_tokens.weight'].data_ptr():
                continue
            missing.append(name)
        elif '.qkv_proj.bias' in name and loaded[name] != {'q', 'k', 'v'}:
            raise ValueError(f'incomplete QKV bias: {name}')
    if missing:
        raise ValueError(f'missing checkpoint parameters: {missing}')
