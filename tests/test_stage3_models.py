"""Opt-in GPU comparison of tiny HF Llama and Qwen3-MoE checkpoints."""
import os
import subprocess
import sys
import pytest


@pytest.mark.skipif(not os.getenv('WWVLLM_STAGE3_GPU'), reason='set WWVLLM_STAGE3_GPU=1')
@pytest.mark.parametrize('kind', ['llama', 'qwen3_moe'])
def test_tiny_model_matches_huggingface(tmp_path, kind):
    if not os.getenv('WWVLLM_STAGE3_CHILD'):
        env = dict(os.environ, WWVLLM_STAGE3_CHILD='1')
        subprocess.run([sys.executable, '-m', 'pytest', '-q', __file__, '-k', kind], env=env, check=True)
        return
    import torch
    from transformers import LlamaConfig, LlamaForCausalLM, Qwen3MoeConfig, Qwen3MoeForCausalLM
    from wwvllm.config import Config
    from wwvllm.engine.model_runner import ModelRunner
    from wwvllm.engine.sequence import Sequence
    from wwvllm.utils.context import get_context, reset_context

    common = dict(vocab_size=128, hidden_size=128, intermediate_size=256,
                  num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                  max_position_embeddings=64, tie_word_embeddings=False)
    if kind == 'llama':
        hf_config = LlamaConfig(**common)
        hf_model = LlamaForCausalLM(hf_config)
    else:
        hf_config = Qwen3MoeConfig(**common, num_experts=4, num_experts_per_tok=2,
                                   moe_intermediate_size=64, norm_topk_prob=True)
        hf_model = Qwen3MoeForCausalLM(hf_config)
    hf_model = hf_model.eval().to(dtype=torch.float16)
    hf_model.save_pretrained(tmp_path, safe_serialization=True)
    ids = torch.tensor([[1, 5, 7, 8, 9]], device='cuda')
    hf_model.cuda()
    with torch.inference_mode():
        expected = hf_model(ids).logits[0].float().cpu()
    del hf_model
    torch.cuda.empty_cache()
    config = Config(str(tmp_path), max_num_batched_tokens=32, max_num_seqs=2,
                    max_model_len=64, gpu_memory_utilization=.4, enforce_eager=True)
    runner = ModelRunner(config, 0, [])
    try:
        seq = Sequence(ids[0].tolist())
        seq.num_scheduled_tokens = len(seq)
        seq.block_table = [0]
        input_ids, positions = runner.prepare_prefill([seq], [True])
        get_context().sample_indices = torch.arange(len(seq), device='cuda')
        with torch.inference_mode():
            actual = runner.run_model(input_ids, positions, True).float().cpu()
        reset_context()
        torch.testing.assert_close(actual, expected, atol=.004, rtol=.04)
    finally:
        runner.exit()


@pytest.mark.skipif(not os.getenv('WWVLLM_STAGE3_GPU'), reason='set WWVLLM_STAGE3_GPU=1')
@pytest.mark.parametrize('method', ['gptq', 'awq'])
def test_packed_checkpoint_matches_dequantized_hf(tmp_path, method):
    if not os.getenv('WWVLLM_STAGE3_CHILD'):
        env = dict(os.environ, WWVLLM_STAGE3_CHILD='1')
        subprocess.run([sys.executable, '-m', 'pytest', '-q', __file__, '-k', method], env=env, check=True)
        return
    import json
    import torch
    from safetensors.torch import save_file
    from transformers import LlamaConfig, LlamaForCausalLM
    from wwvllm.config import Config
    from wwvllm.engine.model_runner import ModelRunner
    from wwvllm.engine.sequence import Sequence
    from wwvllm.utils.context import get_context, reset_context

    def pack(values):
        chunks = values.to(torch.int64).reshape(*values.shape[:-1], -1, 8)
        return ((chunks << (torch.arange(8) * 4)).sum(-1) & 0xffffffff).to(torch.int32)

    hf_config = LlamaConfig(vocab_size=128, hidden_size=128, intermediate_size=256,
                            num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2,
                            max_position_embeddings=64, tie_word_embeddings=False)
    hf = LlamaForCausalLM(hf_config).eval().to(dtype=torch.float16)
    hf.save_pretrained(tmp_path, safe_serialization=True)
    weights = {}
    for name, value in hf.state_dict().items():
        if name.endswith('.weight') and any(f'.{part}.weight' in name for part in
            ('q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj')):
            n, k = value.shape
            grouped = value.float().reshape(n, -1, 32)
            scale = grouped.abs().amax(-1).clamp_min(1e-8) / 7
            q = (grouped / scale[..., None]).round().clamp(-7, 7).to(torch.int32).reshape(n, k) + 8
            dequant = ((q.reshape(n, -1, 32) - 8) * scale[..., None]).reshape(n, k).to(value.dtype)
            value.copy_(dequant)
            group_count = k // 32
            if method == 'gptq':
                qweight = pack(q).T.contiguous()
                qzeros = pack(torch.full((group_count, n), 7))
            else:
                inverse = [0, 2, 4, 6, 1, 3, 5, 7]
                qweight = pack(q.T.contiguous().reshape(k, -1, 8)[..., inverse].reshape(k, n))
                qzeros = pack(torch.full((group_count, n), 8))
            prefix = name[:-len('weight')]
            weights[prefix + 'qweight'] = qweight
            weights[prefix + 'qzeros'] = qzeros
            weights[prefix + 'scales'] = scale.T.contiguous().half()
        else:
            weights[name] = value.contiguous()
    save_file(weights, str(tmp_path / 'model.safetensors'))
    config_file = tmp_path / 'config.json'
    data = json.loads(config_file.read_text())
    data['quantization_config'] = dict(quant_method=method, bits=4, group_size=32, version='gemm', zero_point=True)
    config_file.write_text(json.dumps(data))
    ids = torch.tensor([[1, 5, 7, 8, 9]], device='cuda')
    hf.cuda()
    with torch.inference_mode():
        expected = hf(ids).logits[0].float().cpu()
    del hf
    torch.cuda.empty_cache()
    config = Config(str(tmp_path), max_num_batched_tokens=32, max_num_seqs=2, max_model_len=64,
                    gpu_memory_utilization=.4, enforce_eager=True)
    runner = ModelRunner(config, 0, [])
    try:
        seq = Sequence(ids[0].tolist())
        seq.num_scheduled_tokens = len(seq)
        seq.block_table = [0]
        inputs, positions = runner.prepare_prefill([seq], [True])
        get_context().sample_indices = torch.arange(len(seq), device='cuda')
        with torch.inference_mode():
            actual = runner.run_model(inputs, positions, True).float().cpu()
        reset_context()
        torch.testing.assert_close(actual, expected, atol=.004, rtol=.04)
    finally:
        runner.exit()


@pytest.mark.skipif(not os.getenv('WWVLLM_TEST_MODEL'), reason='set WWVLLM_TEST_MODEL')
def test_int8_target_with_speculative_decode():
    if not os.getenv('WWVLLM_STAGE3_CHILD'):
        env = dict(os.environ, WWVLLM_STAGE3_CHILD='1')
        subprocess.run([sys.executable, '-m', 'pytest', '-q', __file__, '-k', 'int8_target'], env=env, check=True)
        return
    from wwvllm import LLM, SamplingParams
    path = os.environ['WWVLLM_TEST_MODEL']
    engine = LLM(path, quantization='int8', enable_speculative=True, speculative_model=path,
                 speculative_num_tokens=2, enforce_eager=True, max_num_batched_tokens=32,
                 prefill_chunk_size=16, max_model_len=128, max_num_seqs=2, gpu_memory_utilization=.85)
    try:
        params = SamplingParams(temperature=0, max_tokens=6, ignore_eos=True)
        engine.scheduler.enable_speculative = False
        expected = engine.generate([[10, 20, 30, 40]], params, use_tqdm=False)
        engine.scheduler.enable_speculative = True
        assert engine.generate([[10, 20, 30, 40]], params, use_tqdm=False) == expected
    finally:
        engine.exit()


@pytest.mark.skipif(not os.getenv('WWVLLM_MOE_MODEL'), reason='set WWVLLM_MOE_MODEL')
def test_real_moe_checkpoint_generates():
    if not os.getenv('WWVLLM_STAGE3_CHILD'):
        env = dict(os.environ, WWVLLM_STAGE3_CHILD='1')
        subprocess.run([sys.executable, '-m', 'pytest', '-q', __file__, '-k', 'real_moe'], env=env, check=True)
        return
    from wwvllm import LLM, SamplingParams
    llm = LLM(os.environ['WWVLLM_MOE_MODEL'], max_num_batched_tokens=32,
              prefill_chunk_size=16, max_model_len=128, max_num_seqs=2, gpu_memory_utilization=.8)
    try:
        result = llm.generate([[10, 20, 30, 40]], SamplingParams(temperature=0, max_tokens=4), use_tqdm=False)
        assert len(result[0]['token_ids']) >= 1
        assert llm.model_runner.enforce_eager
    finally:
        llm.exit()
