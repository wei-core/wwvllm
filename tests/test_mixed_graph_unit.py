from types import SimpleNamespace
import pytest

from wwvllm.config import Config
from wwvllm.engine.model_runner import ModelRunner


def test_mixed_graph_cache_size_must_be_positive(tmp_path):
    with pytest.raises(ValueError, match='max_mixed_graphs'):
        Config(str(tmp_path), enable_mixed_graph=True, max_mixed_graphs=0)


@pytest.mark.parametrize('changes,expected', [
    ({}, 'graph'),
    ({'world_size': 2}, 'eager'),
    ({'quantization': 'int8'}, 'eager'),
    ({'model_type': 'llama'}, 'eager'),
    ({'enforce_eager': True}, 'eager'),
    ({'enable_mixed_graph': False}, 'eager'),
])
def test_mixed_graph_only_on_supported_paths(changes, expected):
    calls = []
    runner = SimpleNamespace(
        enable_mixed_graph=changes.get('enable_mixed_graph', True),
        world_size=changes.get('world_size', 1),
        config=SimpleNamespace(quantization=changes.get('quantization'),
                               hf_config=SimpleNamespace(model_type=changes.get('model_type', 'qwen3'))),
        enforce_eager=changes.get('enforce_eager', False), rank=1,
        prepare_prefill=lambda seqs, flags: ('ids', 'positions'),
        run_mixed_graph=lambda ids, positions: calls.append('graph'),
        run_model=lambda ids, positions, prefill: calls.append('eager'),
    )
    decode = SimpleNamespace(num_cached_tokens=5, num_prompt_tokens=5,
                             num_tokens=6, num_scheduled_tokens=1)
    prefill = SimpleNamespace(num_cached_tokens=0, num_prompt_tokens=4,
                              num_tokens=4, num_scheduled_tokens=2)
    assert ModelRunner.run(runner, [decode, prefill], True, [True, False]) == []
    assert calls == [expected]
