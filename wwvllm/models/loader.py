"""Streaming safetensors checkpoint reader and strict tensor-to-parameter loader."""
from collections import defaultdict
import json
from pathlib import Path
import torch
from torch import nn
from safetensors import safe_open


def default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor):
    if param.shape != loaded_weight.shape:
        raise ValueError(f'weight shape mismatch: {tuple(loaded_weight.shape)} != {tuple(param.shape)}')
    param.data.copy_(loaded_weight)


class SafeTensorsReader:
    def __init__(self, path):
        self.path = Path(path)
        indices = sorted(self.path.glob('*.safetensors.index.json'))
        if len(indices) > 1:
            raise ValueError('multiple safetensors indices')
        self.weight_map = None
        if indices:
            def no_duplicate_keys(pairs):
                result = {}
                for key, value in pairs:
                    if key in result:
                        raise ValueError(f'duplicate index entry: {key}')
                    result[key] = value
                return result
            with indices[0].open() as f:
                self.weight_map = json.load(f, object_pairs_hook=no_duplicate_keys)['weight_map']
            self.files = [self.path / name for name in sorted(set(self.weight_map.values()))]
        else:
            self.files = sorted(self.path.glob('*.safetensors'))
        if not self.files or any(not p.is_file() for p in self.files):
            raise FileNotFoundError('checkpoint shards are missing')

    def iter_tensors(self):
        seen = set()
        for file in self.files:
            with safe_open(file, framework='pt', device='cpu') as reader:
                for name in reader.keys():
                    if name in seen:
                        raise ValueError(f'duplicate tensor: {name}')
                    if self.weight_map is not None and self.weight_map.get(name) != str(file.relative_to(self.path)):
                        raise ValueError(f'tensor {name} does not match index')
                    seen.add(name)
                    yield name, reader.get_tensor(name)
        if self.weight_map is not None and seen != set(self.weight_map):
            raise ValueError(f'index names absent from shards: {sorted(set(self.weight_map) - seen)}')


def load_model(model: nn.Module, path: str):
    mapping = getattr(model, 'packed_modules_mapping', {})
    params = dict(model.named_parameters(remove_duplicate=False))
    loaded = defaultdict(set)
    for weight_name, tensor in SafeTensorsReader(path).iter_tensors():
        target, shard = weight_name, None
        # Experts use separate unmerged gate/up weights; exact path segments only.
        if '.experts.' not in weight_name:
            parts = weight_name.split('.')
            for i, part in enumerate(parts):
                if part in mapping:
                    replacement, shard = mapping[part]
                    parts[i] = replacement
                    target = '.'.join(parts)
                    break
        if target not in params:
            raise ValueError(f'unexpected checkpoint tensor: {weight_name}')
        if shard in loaded[target]:
            raise ValueError(f'duplicate parameter shard: {target} {shard}')
        param = params[target]
        loader = getattr(param, 'weight_loader', default_weight_loader)
        try:
            loader(param, tensor, shard) if shard is not None else loader(param, tensor)
        except (RuntimeError, AssertionError) as exc:
            raise ValueError(f'invalid tensor {weight_name}: {exc}') from exc
        loaded[target].add(shard)
    missing = []
    for name, param in params.items():
        shards = loaded[name]
        if not shards:
            # HF omits lm_head when embeddings are tied.
            if name == 'lm_head.weight' and 'model.embed_tokens.weight' in loaded and \
               param.data_ptr() == params['model.embed_tokens.weight'].data_ptr():
                continue
            missing.append(name)
        elif '.qkv_proj.' in name and shards != {'q', 'k', 'v'}:
            raise ValueError(f'incomplete QKV shards: {name}: {shards}')
        elif '.gate_up_proj.' in name and shards != {0, 1}:
            raise ValueError(f'incomplete gate/up shards: {name}: {shards}')
    if missing:
        raise ValueError(f'missing checkpoint parameters: {missing}')
