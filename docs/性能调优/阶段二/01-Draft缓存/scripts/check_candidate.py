"""Compare candidate proposals to full recomputation after acceptance and rollback."""
import importlib.util
import torch
from types import SimpleNamespace
from wwvllm.engine.draft_runner import DraftRunner
original = DraftRunner.propose
spec = importlib.util.spec_from_file_location('candidate', 'docs/性能调优/阶段二/01-Draft缓存/scripts/candidate.py')
module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
model='/home/ubuntu/ai-infra/minivllm/Models/Qwen3-0.6B'
runner=DraftRunner(model, torch.bfloat16)
seq=SimpleNamespace(seq_id=1,token_ids=[10,20,30,40],temperature=0,top_k=-1,top_p=1.0)
for accepted in (4,0,2,1,3,4):
    baseline, bp = original(runner, seq, 4)
    candidate, cp = module.propose(runner, seq, 4)
    assert baseline == candidate, (baseline,candidate)
    for a,b in zip(bp,cp): torch.testing.assert_close(a,b,atol=0.002,rtol=0.002)
    seq.token_ids.extend(baseline[:accepted]+[55+accepted])
# A second request must not contaminate the first.
other=SimpleNamespace(seq_id=2,token_ids=[1,2,3],temperature=0,top_k=-1,top_p=1.0)
assert original(runner,other,4)[0] == module.propose(runner,other,4)[0]
assert original(runner,seq,4)[0] == module.propose(runner,seq,4)[0]
print('Draft cache acceptance, rejection and request isolation: PASS')
