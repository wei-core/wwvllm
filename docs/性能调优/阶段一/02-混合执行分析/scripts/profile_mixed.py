"""Diagnostic profile of one warmed, mixed Qwen3 step (not a latency benchmark)."""
import atexit
import random
import torch
from torch.profiler import profile, ProfilerActivity
from wwvllm import LLM, SamplingParams

MODEL = '/home/ubuntu/ai-infra/minivllm/Models/Qwen3-0.6B'
llm = LLM(MODEL, max_num_batched_tokens=512, max_num_seqs=8,
          max_model_len=2048, gpu_memory_utilization=.60,
          enforce_eager=False, tensor_parallel_size=1)


def prepare(seed):
    rng = random.Random(seed)
    first = [rng.randint(1, 10000) for _ in range(64)]
    prompts = [[rng.randint(1, 10000) for _ in range(1024)] for _ in range(2)]
    llm.add_request(first, SamplingParams(temperature=.6, max_tokens=64, ignore_eos=True))
    for _ in range(4):
        llm.step()
    for prompt in prompts:
        llm.add_request(prompt, SamplingParams(temperature=.6, max_tokens=16, ignore_eos=True))


try:
    prepare(100)
    while not llm.is_finished():
        llm.step()
    prepare(101)
    schedule = llm.scheduler.schedule
    observed = {}
    def trace():
        output = schedule()
        observed.update(mixed=output.has_decode and output.has_prefill,
                        tokens=output.num_scheduled_tokens,
                        requests=[{'mode': x.mode.name, 'tokens': x.num_scheduled_tokens} for x in output.scheduled])
        return output
    llm.scheduler.schedule = trace
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=False) as prof:
        llm.step()
        torch.cuda.synchronize()
    print('schedule:', observed)
    print('\nCPU operators (profiler timing, not benchmark):')
    print(prof.key_averages().table(sort_by='self_cpu_time_total', row_limit=25))
    print('\nCUDA operators (profiler timing, not benchmark):')
    print(prof.key_averages().table(sort_by='self_device_time_total', row_limit=25))
finally:
    atexit.unregister(llm.exit)
    llm.exit()
