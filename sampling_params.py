from dataclasses import dataclass


@dataclass(slots=True)
class SamplingParams:
    temperature: float = 1.0
    max_tokens: int = 64
    ignore_eos: bool = False
    top_k: int = 0
    top_p: float = 1.0
    speculative: bool = True

    def __post_init__(self):
        if self.temperature < 0 or self.max_tokens < 1 or self.top_k < 0 or not 0 < self.top_p <= 1:
            raise ValueError("invalid temperature, max_tokens, top_k, or top_p")
