"""Run one isolated arm: ordinary decode, draft rebuild, or experimental draft reuse."""
import argparse, atexit, importlib.util, json, os, sys
from pathlib import Path
import torch
from transformers import AutoConfig, AutoTokenizer

ROOT = Path(__file__).resolve().parents[5]
p=argparse.ArgumentParser()
p.add_argument('--target',default='/home/ubuntu/ai-infra/minivllm/Models/Qwen3-1.7B')
p.add_argument('--draft',default='/home/ubuntu/ai-infra/minivllm/Models/Qwen3-0.6B')
p.add_argument('--mode',choices=('normal','rebuild','cache'),required=True)
p.add_argument('--output',required=True)
p.add_argument('--repeats',type=int,default=2)
p.add_argument('--gpu-memory-utilization',type=float,default=.96)
p.add_argument('--draft-memory-reserve',type=float,default=.04)
args=p.parse_args()

def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path); mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod

if args.mode == 'cache':
    load('candidate',ROOT/'docs/性能调优/阶段二/01-Draft缓存/scripts/candidate.py')
bench=load('stage2_compare',ROOT/'docs/性能对照/阶段二/scripts/three_way_compare.py')
bench.CASES={'short_batch':[(128,32)]*4,'long_batch':[(512,16)]*2,
             'staggered':[(64,64),(384,16),(384,16)]}
from wwvllm import LLM, SamplingParams
import wwvllm

a,b=[AutoTokenizer.from_pretrained(p) for p in (args.target,args.draft)]
configs=[AutoConfig.from_pretrained(p) for p in (args.target,args.draft)]
compat=dict(vocab_equal=a.get_vocab()==b.get_vocab(),special_ids_equal=a.all_special_ids==b.all_special_ids,
            special_tokens_map_equal=a.special_tokens_map==b.special_tokens_map,
            config_special_equal=(configs[0].bos_token_id,configs[0].eos_token_id)==(configs[1].bos_token_id,configs[1].eos_token_id),
            vocab_length=len(a.get_vocab()))
if not all(compat[k] for k in ('vocab_equal','special_ids_equal','special_tokens_map_equal','config_special_equal')):
    raise ValueError(f'incompatible tokenizer: {compat}')
options=dict(max_num_batched_tokens=256,max_num_seqs=4,max_model_len=1024,prefill_chunk_size=256,
             gpu_memory_utilization=args.gpu_memory_utilization,draft_memory_reserve=args.draft_memory_reserve,
             enforce_eager=True,tensor_parallel_size=1,enable_speculative=True,
             speculative_model=args.draft,speculative_num_tokens=4)
free_before,total=torch.cuda.mem_get_info(); llm=LLM(args.target,**options)
if args.mode=='normal': llm.scheduler.enable_speculative=False
rows=[]
try:
    for case in bench.CASES: bench.workload(llm,SamplingParams,'stage2',case,-1)
    torch.cuda.reset_peak_memory_stats()
    for repeat in range(args.repeats):
        for case in bench.CASES:
            row=bench.workload(llm,SamplingParams,'stage2',case,repeat)
            rows.append(row)
            print(json.dumps({k:v for k,v in row.items() if k!='request_token_offsets_s'}),flush=True)
    data=dict(mode=args.mode,target=args.target,draft=args.draft,tokenizer=compat,config=options,cases=bench.CASES,
              gpu=torch.cuda.get_device_name(),torch=torch.__version__,package_path=wwvllm.__file__,
              sampling={'temperature':.6,'ignore_eos':True,'prompt':'uniform token ids 1..10000'},
              gpu_free_before_bytes=free_before,gpu_total_bytes=total,target_kv_blocks=llm.config.num_kvcache_blocks,
              peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),results=rows)
    with open(args.output,'w') as f: json.dump(data,f,indent=2)
finally:
    atexit.unregister(llm.exit); llm.exit()
