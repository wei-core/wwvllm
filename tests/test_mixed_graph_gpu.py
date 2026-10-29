"""Opt-in mixed CUDA graph correctness test using a local Qwen3 checkpoint."""
import os
import subprocess
import sys
import pytest


@pytest.mark.skipif(not os.getenv('WWVLLM_TEST_MODEL'), reason='set WWVLLM_TEST_MODEL')
def test_mixed_graph_matches_eager_and_handles_eviction():
    if not os.getenv('WWVLLM_MIXED_GRAPH_CHILD'):
        env = dict(os.environ, WWVLLM_MIXED_GRAPH_CHILD='1')
        subprocess.run([sys.executable, '-m', 'pytest', '-q', __file__], env=env, check=True)
        return
    import torch
    from wwvllm import LLM, SamplingParams
    from wwvllm.utils.context import reset_context

    llm = LLM(os.environ['WWVLLM_TEST_MODEL'], enable_mixed_graph=True, max_mixed_graphs=2,
              max_num_batched_tokens=64, prefill_chunk_size=16, max_model_len=512,
              max_num_seqs=2, gpu_memory_utilization=.60)
    params = SamplingParams(temperature=0, max_tokens=6, ignore_eos=True)
    prompts = [[10, 20, 30, 40], list(range(1, 258))]
    try:
        llm.model_runner.enable_mixed_graph = False
        expected = llm.generate(prompts, params, use_tqdm=False)
        # Clear reusable prefix hashes so the same prompts still exercise mixed
        # chunks instead of immediately reusing the entire first full block.
        llm.scheduler.block_manager.hash_to_block_id.clear()
        llm.model_runner.enable_mixed_graph = True
        actual = llm.generate(prompts, params, use_tqdm=False)
        assert [x['token_ids'] for x in actual] == [x['token_ids'] for x in expected]
        stats = llm.model_runner.mixed_graph_stats
        assert stats['captures'] >= 3 and stats['replays'] >= stats['captures']
        assert stats['evictions'] > 0 and len(llm.model_runner.mixed_graphs) <= 2

        # Compare all sampled logits for one actual mixed batch, reusing its KV
        # slots. Capture/replay and eager use identical inputs and metadata.
        llm.scheduler.block_manager.hash_to_block_id.clear()
        llm.add_request(prompts[0], params)
        llm.add_request(prompts[1], params)
        llm.step()
        output = llm.scheduler.schedule()
        assert output.has_decode and output.has_prefill
        seqs = [item.seq for item in output.scheduled]
        flags = [item.should_sample for item in output.scheduled]
        with torch.inference_mode():
            ids, positions = llm.model_runner.prepare_prefill(seqs, flags)
            graph_logits = llm.model_runner.run_mixed_graph(ids, positions).clone()
            eager_logits = llm.model_runner.run_model(ids, positions, True)
            torch.testing.assert_close(graph_logits.float(), eager_logits.float(), atol=.01, rtol=.01)
        reset_context()
    finally:
        llm.exit()
