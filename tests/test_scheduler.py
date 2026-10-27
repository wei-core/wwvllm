from types import SimpleNamespace
import pickle

import pytest
from wwvllm.engine.sequence import Sequence, SequenceStatus
from wwvllm.engine.scheduler import Scheduler
from wwvllm.sampling_params import SamplingParams


def scheduler(budget=4, chunk=3, blocks=16, max_seqs=4):
    return Scheduler(SimpleNamespace(max_num_seqs=max_seqs, max_num_batched_tokens=budget,
        prefill_chunk_size=chunk, eos=-1, kvcache_block_size=4, num_kvcache_blocks=blocks))


@pytest.fixture(autouse=True)
def block_size():
    old = Sequence.block_size
    Sequence.block_size = 4
    yield
    Sequence.block_size = old


def test_chunking_and_mixed_sampling():
    s = scheduler()
    a = Sequence([1, 2], SamplingParams(temperature=0, max_tokens=3))
    s.add(a)
    out = s.schedule()
    s.postprocess(out, [7])
    b = Sequence(list(range(10)))
    s.add(b)
    out = s.schedule()
    assert out.has_decode and out.has_prefill and out.num_scheduled_tokens == 4
    assert [i.should_sample for i in out.scheduled] == [True, False]
    s.postprocess(out, [8])
    assert a.completion_token_ids == [7, 8]
    assert b.num_cached_tokens == 3 and b.completion_token_ids == []
    assert len(b.block_table) == 1
    assert pickle.loads(pickle.dumps(b)).token_ids == b.token_ids


def test_prefix_reuse_and_release():
    s = scheduler(budget=8, chunk=8)
    a = Sequence(list(range(8)), SamplingParams(max_tokens=1))
    s.add(a)
    s.postprocess(s.schedule(), [20])
    b = Sequence(list(range(8)), SamplingParams(max_tokens=1))
    s.add(b)
    out = s.schedule()
    assert b.num_cached_tokens == 4
    assert out.num_scheduled_tokens == 4
    s.postprocess(out, [21])
    assert len(s.block_manager.free_block_ids) == 16


def test_cancel_and_capacity():
    s = scheduler(blocks=2)
    with pytest.raises(ValueError):
        s.add(Sequence(list(range(9))))
    seq = Sequence(list(range(6)))
    s.add(seq)
    s.postprocess(s.schedule(), [])
    assert s.cancel(seq) and seq.status == SequenceStatus.CANCELLED
    assert len(s.block_manager.free_block_ids) == 2
    assert not s.cancel(seq)


def test_preemption_recomputes_without_duplicate_output():
    s = scheduler(budget=8, chunk=4, blocks=2, max_seqs=2)
    a = Sequence([1, 2, 3, 4], SamplingParams(max_tokens=3))
    b = Sequence([5, 6, 7, 8], SamplingParams(max_tokens=3))
    s.add(a)
    s.add(b)
    # Both prompts fill the cache; decode needs another block.
    s.postprocess(s.schedule(), [10, 11])
    out = s.schedule()
    assert a.preemption_count + b.preemption_count >= 1
    assert all(item.seq.num_cached_tokens + item.num_scheduled_tokens <= item.seq.num_tokens for item in out.scheduled)
    s.postprocess(out, [42] * sum(item.should_sample for item in out.scheduled))
    for _ in range(20):
        if s.is_finished():
            break
        out = s.schedule()
        s.postprocess(out, [42] * sum(item.should_sample for item in out.scheduled))
    assert s.is_finished()
    assert len(a.completion_token_ids) == len(b.completion_token_ids) == 3
    assert len(s.block_manager.free_block_ids) == 2
