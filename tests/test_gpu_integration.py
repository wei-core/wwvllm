"""Opt-in real-model test: WWVLLM_TEST_MODEL=/path/to/model pytest tests/test_gpu_integration.py."""
import os
import pytest


@pytest.mark.skipif(not os.getenv('WWVLLM_TEST_MODEL'), reason='set WWVLLM_TEST_MODEL to a local Qwen3 model')
def test_mixed_matches_unchunked_with_decode_graph():
    from wwvllm import LLM, SamplingParams
    llm = LLM(os.environ['WWVLLM_TEST_MODEL'], max_num_batched_tokens=128,
              prefill_chunk_size=128, max_model_len=256, max_num_seqs=4,
              gpu_memory_utilization=0.45)
    try:
        prompts = [[10, 20, 30, 40], list(range(1, 70))]
        params = SamplingParams(temperature=0, max_tokens=5, ignore_eos=True)
        reference = llm.generate(prompts, params, use_tqdm=False)
        llm.scheduler.chunk_size = 8
        actual = llm.generate(prompts, params, use_tqdm=False)
        assert [x['token_ids'] for x in actual] == [x['token_ids'] for x in reference]
    finally:
        llm.exit()
