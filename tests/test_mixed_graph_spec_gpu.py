"""Exercise mixed graph replay alongside speculative verification on one GPU."""
import os
import subprocess
import sys
import pytest


@pytest.mark.skipif(not os.getenv('WWVLLM_TEST_MODEL'), reason='set WWVLLM_TEST_MODEL')
def test_mixed_graph_and_speculative_requests_coexist():
    if not os.getenv('WWVLLM_MIXED_SPEC_CHILD'):
        env = dict(os.environ, WWVLLM_MIXED_SPEC_CHILD='1')
        subprocess.run([sys.executable, '-m', 'pytest', '-q', __file__], env=env, check=True)
        return
    from wwvllm import LLM, SamplingParams
    model = os.environ['WWVLLM_TEST_MODEL']
    llm = LLM(model, enable_mixed_graph=True, enable_speculative=True,
              speculative_model=model, speculative_num_tokens=2,
              max_num_batched_tokens=64, prefill_chunk_size=16,
              max_model_len=256, max_num_seqs=4, gpu_memory_utilization=.85)
    prompts = [[10, 20, 30, 40], list(range(1, 100)), [23, 24, 25]]
    params = [SamplingParams(temperature=0, max_tokens=5, ignore_eos=True, speculative=False),
              SamplingParams(temperature=0, max_tokens=5, ignore_eos=True),
              SamplingParams(temperature=0, max_tokens=5, ignore_eos=True)]
    try:
        llm.model_runner.enable_mixed_graph = False
        expected = llm.generate(prompts, params, use_tqdm=False)
        llm.scheduler.block_manager.hash_to_block_id.clear()
        llm.model_runner.enable_mixed_graph = True
        actual = llm.generate(prompts, params, use_tqdm=False)
        assert [x['token_ids'] for x in actual] == [x['token_ids'] for x in expected]
        assert llm.model_runner.mixed_graph_stats['replays'] > 0
    finally:
        llm.exit()
