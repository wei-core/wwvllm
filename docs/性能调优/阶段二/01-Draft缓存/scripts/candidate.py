"""Experimental bounded Draft cache. Run the stage-two comparison with this wrapper."""
import runpy
from collections import OrderedDict
import torch
from wwvllm.engine.draft_runner import DraftRunner
from wwvllm.layers.speculative_sampler import probabilities, sample

original_init = DraftRunner.__init__
def init(self, *args, **kwargs):
    original_init(self, *args, **kwargs)
    self._experiment_cache = OrderedDict()

@torch.inference_mode()
def propose(self, seq, count):
    entries = self._experiment_cache
    previous = entries.pop(seq.seq_id, None)
    cache, prefix = None, 0
    if previous:
        cache, tokens = previous
        for a, b in zip(tokens, seq.token_ids):
            if a != b:
                break
            prefix += 1
        prefix = min(prefix, len(seq.token_ids) - 1)
        cache.crop(prefix)
    ids = torch.tensor([seq.token_ids[prefix:]], device='cuda', dtype=torch.long)
    processed = list(seq.token_ids)
    proposed, distributions = [], []
    for i in range(count):
        output = self.model(input_ids=ids, past_key_values=cache, use_cache=True, logits_to_keep=1)
        cache = output.past_key_values
        probs = probabilities(output.logits[0, -1], seq.temperature, seq.top_k, seq.top_p)
        token = int(probs.argmax()) if seq.temperature == 0 else sample(probs)
        proposed.append(token)
        distributions.append(probs)
        if i + 1 < count:
            processed.append(token)
        ids = torch.tensor([[token]], device='cuda', dtype=torch.long)
    entries[seq.seq_id] = (cache, processed)
    while len(entries) > 8:
        entries.popitem(last=False)
    return proposed, distributions

DraftRunner.__init__ = init
DraftRunner.propose = propose
if __name__ == '__main__':
    runpy.run_path('docs/性能对照/阶段二/scripts/three_way_compare.py', run_name='__main__')
