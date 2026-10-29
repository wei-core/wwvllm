"""Verify greedy token equality for different-sized target and draft models."""
import argparse, importlib.util, json
from pathlib import Path
from wwvllm import LLM, SamplingParams
p=argparse.ArgumentParser()
p.add_argument('--target',default='/home/ubuntu/ai-infra/minivllm/Models/Qwen3-1.7B')
p.add_argument('--draft',default='/home/ubuntu/ai-infra/minivllm/Models/Qwen3-0.6B')
p.add_argument('--cache',action='store_true')
a=p.parse_args()
if a.cache:
    path=Path(__file__).resolve().parents[2]/'01-Draft缓存/scripts/candidate.py'
    spec=importlib.util.spec_from_file_location('candidate',path); module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
llm=LLM(a.target,enable_speculative=True,speculative_model=a.draft,speculative_num_tokens=4,
        max_num_batched_tokens=256,prefill_chunk_size=256,max_num_seqs=4,max_model_len=1024,
        enforce_eager=True,gpu_memory_utilization=.96,draft_memory_reserve=.04)
try:
    prompts=[[10,20,30,40],list(range(1,129))]
    params=SamplingParams(temperature=0,max_tokens=16,ignore_eos=True)
    records=[]
    for label, group in [('single_short',prompts[:1]),('single_long',prompts[1:]),('batch',prompts)]:
        llm.scheduler.enable_speculative=False
        llm.scheduler.block_manager.hash_to_block_id.clear()
        expected=llm.generate(group,params,use_tqdm=False)
        llm.scheduler.enable_speculative=True
        llm.scheduler.block_manager.hash_to_block_id.clear()
        actual=llm.generate(group,params,use_tqdm=False)
        left=[x['token_ids'] for x in expected]; right=[x['token_ids'] for x in actual]
        records.append(dict(case=label,equal=left==right,normal=left,speculative=right))
    print(json.dumps(records),flush=True)
    if not all(r['equal'] for r in records):
        raise SystemExit('greedy equality FAIL')
    print('greedy equality PASS', 'cache' if a.cache else 'rebuild')

finally:
    llm.exit()
