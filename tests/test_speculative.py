from types import SimpleNamespace

import pytest
import torch
from wwvllm.engine.block_manager import BlockManager
from wwvllm.engine.sequence import Sequence
from wwvllm.engine.scheduler import Scheduler
from wwvllm.engine.scheduler_output import ExecutionMode
from wwvllm.layers.speculative_sampler import probabilities, verify
from wwvllm.sampling_params import SamplingParams


def test_probability_filters_and_greedy():
    logits = torch.tensor([4., 3., 2., 1.])
    p = probabilities(logits, 1, top_k=2, top_p=.8)
    assert torch.isclose(p.sum(), torch.tensor(1.))
    assert p[2] == p[3] == 0
    assert probabilities(logits, 0).argmax() == 0
    assert (probabilities(torch.ones(4), 1, top_k=2) > 0).sum() == 2
    with pytest.raises(ValueError):
        SamplingParams(top_p=0)


def test_accept_reject_and_bonus():
    q = torch.tensor([.2, .8])
    p = torch.tensor([.9, .1])
    result = verify([1], [q], [p, p], greedy=True)
    assert result.token_ids == [0] and result.accepted == 0
    result = verify([0, 0], [p, p], [p, p, p], greedy=True)
    assert result.token_ids == [0, 0, 0] and result.accepted == 2


def test_stochastic_sampling_matches_target_distribution():
    torch.manual_seed(1234)
    p = torch.tensor([.6, .3, .1])
    q = torch.tensor([.1, .2, .7])
    counts = torch.zeros(3)
    for _ in range(12000):
        candidate = torch.multinomial(q, 1).item()
        token = verify([candidate], [q], [p, p]).token_ids[0]
        counts[token] += 1
    assert torch.all(torch.abs(counts / counts.sum() - p) < .02)


def test_transaction_across_block_boundary_and_double_close():
    original = Sequence.block_size
    Sequence.block_size = 4
    try:
        manager = BlockManager(4, 4)
        seq = Sequence([1, 2, 3, 4, 5])
        assert manager.reserve(seq, 5)
        seq.num_cached_tokens = 4
        old = list(seq.block_table)
        tx = manager.begin_transaction(seq, 13)
        assert len(seq.block_table) == 4
        with pytest.raises(RuntimeError):
            manager.begin_transaction(seq, 5)
        manager.finish_transaction(tx, 6)
        assert seq.block_table == old
        assert seq.num_cached_tokens == 4
        with pytest.raises(RuntimeError):
            manager.rollback_transaction(tx)
        manager.deallocate(seq)
        assert len(manager.free_block_ids) == 4
    finally:
        Sequence.block_size = original


def test_scheduler_spec_budget_commit_and_fallback():
    old = Sequence.block_size
    Sequence.block_size = 4
    try:
        config = SimpleNamespace(max_num_seqs=2, max_num_batched_tokens=4,
            prefill_chunk_size=4, eos=-1, kvcache_block_size=4, num_kvcache_blocks=8,
            enable_speculative=True, speculative_num_tokens=3)
        s = Scheduler(config)
        seq = Sequence([1, 2, 3], SamplingParams(temperature=0, max_tokens=5))
        s.add(seq)
        s.postprocess(s.schedule(), [4])
        out = s.schedule()
        item = out.scheduled[0]
        assert item.mode == ExecutionMode.SPECULATIVE and out.num_scheduled_tokens == 4
        from wwvllm.layers.speculative_sampler import SpeculativeResult
        events = s.postprocess(out, [], {seq.seq_id: SpeculativeResult([5, 6], 1, 3)})
        assert [e[1] for e in events] == [5, 6]
        assert seq.num_cached_tokens == seq.num_tokens - 1
        assert seq.completion_token_ids == [4, 5, 6]
        seq.speculative_enabled = False
        assert s.schedule().scheduled[0].mode == ExecutionMode.DECODE
    finally:
        Sequence.block_size = old


def test_first_and_later_rejection_and_bonus_stochastic(monkeypatch):
    # Fixed uniform draws exercise rejection at both positions and full acceptance.
    p0 = torch.tensor([.8, .2])
    q0 = torch.tensor([.2, .8])
    draws = iter([.9, .1, .9])
    monkeypatch.setattr(torch, 'rand', lambda *args, **kwargs: torch.tensor(next(draws)))
    first = verify([1], [q0], [p0, p0])
    assert first.accepted == 0 and first.token_ids == [0]
    # First candidate has ratio 1, second candidate is rejected.
    second = verify([0, 1], [p0, q0], [p0, p0, p0])
    assert second.accepted == 1 and second.token_ids == [0, 0]


def test_eos_truncates_accepted_proposals():
    from wwvllm.layers.speculative_sampler import SpeculativeResult
    old = Sequence.block_size
    Sequence.block_size = 4
    try:
        config = SimpleNamespace(max_num_seqs=1, max_num_batched_tokens=4, prefill_chunk_size=4,
                                 eos=9, kvcache_block_size=4, num_kvcache_blocks=4,
                                 enable_speculative=True, speculative_num_tokens=3)
        s = Scheduler(config)
        seq = Sequence([1, 2, 3], SamplingParams(max_tokens=10))
        s.add(seq)
        s.postprocess(s.schedule(), [4])
        out = s.schedule()
        events = s.postprocess(out, [], {seq.seq_id: SpeculativeResult([5, 9, 7, 8], 3, 3)})
        assert [token for _, token, _ in events] == [5, 9]
        assert events[-1][2] and seq.completion_token_ids == [4, 5, 9]
        assert len(s.block_manager.free_block_ids) == 4
    finally:
        Sequence.block_size = old
