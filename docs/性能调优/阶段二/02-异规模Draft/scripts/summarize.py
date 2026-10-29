"""Summarize separate process JSON files, preserving run and acceptance counts."""
import json, statistics
from pathlib import Path
root=Path(__file__).resolve().parents[1]/'data'
for name,files in (('normal',['normal-1.json']),('rebuild',['rebuild-1.json','rebuild-2.json']),('cache',['cache-1.json','cache-2.json'])):
    datasets=[json.loads((root/f).read_text()) for f in files]
    rows=[r for d in datasets for r in d['results']]
    print(name, 'KV blocks', [d['target_kv_blocks'] for d in datasets], 'allocated peak bytes', [d['peak_allocated_bytes'] for d in datasets])
    for case in ('short_batch','long_batch','staggered'):
        group=[r for r in rows if r['case']==case]
        means={k:statistics.mean(r[k] for r in group) for k in ('output_tokens_per_s','ttft_p50_s','tpot_p50_s','itl_p95_s','target_forwards','verification_forwards')}
        n=sum(r['proposed'] for r in group); a=sum(r['accepted'] for r in group)
        print(case,'n=',len(group),'accepted=',f'{a}/{n}',json.dumps(means,ensure_ascii=False))
