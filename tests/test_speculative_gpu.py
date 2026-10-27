"""Set WWVLLM_TEST_MODEL to enable the single-GPU speculative integration test."""
import os
import pytest


@pytest.mark.skipif(not os.getenv('WWVLLM_TEST_MODEL'), reason='set WWVLLM_TEST_MODEL')
def test_gpu_speculation_matches_normal_and_forced_rejection():
    import torch
    from wwvllm import LLM, SamplingParams
    model = os.environ['WWVLLM_TEST_MODEL']
    llm = LLM(model, enable_speculative=True, speculative_model=model, speculative_num_tokens=2,
              enforce_eager=True, max_num_batched_tokens=128, prefill_chunk_size=64,
              max_model_len=512, max_num_seqs=2, gpu_memory_utilization=.85)
    try:
        prompts = [[10, 20, 30, 40], list(range(1, 256))]
        params = SamplingParams(temperature=0, max_tokens=7, ignore_eos=True)
        llm.scheduler.enable_speculative = False
        expected = llm.generate(prompts, params, use_tqdm=False)
        llm.scheduler.enable_speculative = True
        actual = llm.generate(prompts, params, use_tqdm=False)
        assert actual == expected
        # Exercise probability correction with real logits and top-k/top-p.
        random_output = llm.generate([[10, 20, 30]], SamplingParams(
            temperature=.8, top_k=20, top_p=.9, max_tokens=5, ignore_eos=True), use_tqdm=False)
        assert len(random_output[0]['token_ids']) == 5
        # Force mismatches at every candidate, so correction and KV rollback execute.
        def wrong(seq, count):
            return [151900] * count, [torch.empty(0, device='cuda')] * count
        llm.model_runner.draft_runner.propose = wrong
        rejected = llm.generate(prompts, params, use_tqdm=False)
        assert rejected == expected
    finally:
        llm.exit()
