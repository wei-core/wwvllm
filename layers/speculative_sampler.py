"""Distribution-preserving speculative sampling."""
from dataclasses import dataclass
import torch


@dataclass
class SpeculativeResult:
    token_ids: list[int]
    accepted: int
    proposed: int


def probabilities(logits, temperature, top_k=0, top_p=1.0):
    logits = logits.float()
    if temperature == 0:
        return torch.nn.functional.one_hot(logits.argmax(dim=-1), logits.size(-1)).float()
    logits = logits / temperature
    if top_k:
        k = min(top_k, logits.size(-1))
        indices = logits.topk(k, dim=-1).indices
        mask = torch.ones_like(logits, dtype=torch.bool).scatter(-1, indices, False)
        logits = logits.masked_fill(mask, -torch.inf)
    if top_p < 1:
        sorted_logits, indices = torch.sort(logits, dim=-1, descending=True)
        cumulative = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
        remove = cumulative - torch.softmax(sorted_logits, dim=-1) >= top_p
        sorted_logits = sorted_logits.masked_fill(remove, -torch.inf)
        logits = torch.full_like(logits, -torch.inf).scatter(-1, indices, sorted_logits)
    return torch.softmax(logits, dim=-1)


def sample(probs):
    return torch.multinomial(probs, 1).item()


def verify(draft_ids, draft_probs, target_probs, greedy=False):
    """Accept a prefix and sample a correction or a bonus from target."""
    accepted = []
    for i, token in enumerate(draft_ids):
        p, q = target_probs[i], draft_probs[i]
        if greedy:
            target_token = int(p.argmax())
            if token != target_token:
                return SpeculativeResult(accepted + [target_token], len(accepted), len(draft_ids))
        else:
            q_token = q[token]
            if q_token <= 0:
                raise ValueError("draft candidate has zero probability")
            if torch.rand((), device=p.device) >= torch.clamp(p[token] / q_token, max=1):
                residual = (p - q).clamp_min(0)
                total = residual.sum()
                if total <= 0:
                    raise RuntimeError("empty residual distribution after rejection")
                return SpeculativeResult(accepted + [sample(residual / total)], len(accepted), len(draft_ids))
        accepted.append(token)
    bonus = int(target_probs[-1].argmax()) if greedy else sample(target_probs[-1])
    return SpeculativeResult(accepted + [bonus], len(accepted), len(draft_ids))
