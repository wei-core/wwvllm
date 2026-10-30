"""A small paged-attention inference engine."""


def __getattr__(name):
    if name == "LLM":
        from .llm import LLM
        return LLM
    if name == "AsyncLLMEngine":
        from .engine.async_engine import AsyncLLMEngine
        return AsyncLLMEngine
    if name == "SamplingParams":
        from .sampling_params import SamplingParams
        return SamplingParams
    raise AttributeError(name)
