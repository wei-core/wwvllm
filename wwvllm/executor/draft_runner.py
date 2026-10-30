"""Independent, ephemeral draft KV cache for one proposal at a time."""
import torch
from transformers import AutoModelForCausalLM
from wwvllm.layers.speculative_sampler import probabilities, sample


class DraftRunner:
    def __init__(self, model_path, dtype):
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, dtype=dtype, attn_implementation='eager').eval().cuda()

    @torch.inference_mode()
    def propose(self, seq, count):
        # Rebuild from committed tokens each round. Its KV tensors are independent
        # of the target's paged cache and discarded after verification.
        ids = torch.tensor([seq.token_ids], device='cuda', dtype=torch.long)
        cache = None
        proposed, distributions = [], []
        for _ in range(count):
            output = self.model(input_ids=ids, past_key_values=cache, use_cache=True, logits_to_keep=1)
            cache = output.past_key_values
            probs = probabilities(output.logits[0, -1], seq.temperature, seq.top_k, seq.top_p)
            token = int(probs.argmax()) if seq.temperature == 0 else sample(probs)
            proposed.append(token)
            distributions.append(probs)
            ids = torch.tensor([[token]], device='cuda', dtype=torch.long)
        return proposed, distributions
