"""Compare target logits at shared prefixes in normal and speculative paths.

Diagnostic only: copies GPU tensors to CPU and perturbs timing.
"""
import json
import torch
from wwvllm import LLM, SamplingParams
model='/home/ubuntu/ai-infra/minivllm/Models/Qwen3-1.7B'
draft='/home/ubuntu/ai-infra/minivllm/Models/Qwen3-0.6B'
llm=LLM(model,enable_speculative=True,speculative_model=draft,speculative_num_tokens=4,
        max_num_batched_tokens=256,prefill_chunk_size=256,max_num_seqs=4,max_model_len=1024,
        enforce_eager=True,gpu_memory_utilization=.96,draft_memory_reserve=.04)
runner=llm.model_runner
orig_call,orig_model=runner.call,runner.run_model
current={}; traces={'normal':{},'speculative':{}}

def call(name,*args):
    current.clear();current['name']=name
    if name=='run': current['seqs']=args[0];current['flags']=args[2]
    elif name=='run_speculative': current['seq']=args[0]
    return orig_call(name,*args)

def run_model(ids,positions,prefill):
    result=orig_model(ids,positions,prefill)
    name=current.get('name')
    if name=='run':
        rows=[(tuple(s.token_ids),row) for s,flag,row in zip([s for s,f in zip(current['seqs'],current['flags']) if f],
                                                             [True]*len(result),result)]
        mode='normal'
    elif name=='run_speculative':
        seq=current['seq']; tokens=ids.tolist(); assert tokens[0]==seq.token_ids[-1]
        rows=[(tuple(seq.token_ids+tokens[1:i+1]),row) for i,row in enumerate(result)]
        mode='speculative'
    else: return result
    for key,row in rows:
        vals,idx=torch.topk(row.float(),5)
        traces[mode][key]={'top_ids':idx.tolist(),'top_logits':vals.tolist(),
                           'logit_2578':float(row[2578]),'logit_5646':float(row[5646])}
    return result
runner.call=call;runner.run_model=run_model
try:
    params=SamplingParams(temperature=0,max_tokens=16,ignore_eos=True)
    llm.scheduler.enable_speculative=False
    normal=llm.generate([list(range(1,129))],params,use_tqdm=False)[0]['token_ids']
    llm.scheduler.block_manager.hash_to_block_id.clear()
    llm.scheduler.enable_speculative=True
    spec=llm.generate([list(range(1,129))],params,use_tqdm=False)[0]['token_ids']
    common=sorted(traces['normal'].keys() & traces['speculative'].keys(),key=len)
    result=[dict(prefix_len=len(key),normal=traces['normal'][key],speculative=traces['speculative'][key])
            for key in common if traces['normal'][key]['top_ids'][0]!=traces['speculative'][key]['top_ids'][0]]
    print(json.dumps(dict(normal_tokens=normal,speculative_tokens=spec,shared_prefixes=len(common),different_argmax=result),indent=2),flush=True)
finally:
    runner.call=orig_call;runner.run_model=orig_model;llm.exit()
