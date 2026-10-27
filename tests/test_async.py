import asyncio
from time import sleep
from types import SimpleNamespace

from wwvllm.engine.async_llm_engine import AsyncLLMEngine
from wwvllm.engine import llm_engine
from wwvllm.engine.sequence import Sequence
from wwvllm.sampling_params import SamplingParams


class FakeEngine:
    def __init__(self, *args, **kwargs):
        self.seqs = []
        self.scheduler = self
        self.tokenizer = SimpleNamespace(decode=lambda ids: str(ids))

    def add_request(self, prompt, params):
        seq = Sequence(prompt, params)
        self.seqs.append(seq)
        return seq

    def cancel(self, seq):
        self.seqs.remove(seq)

    def step(self):
        sleep(0.01)
        self.last_events = []
        for seq in self.seqs[:]:
            seq.append_token(42)
            seq.token_timestamps.append(1.0)
            done = seq.num_completion_tokens == seq.max_tokens
            self.last_events.append((seq, 42, done))
            if done:
                self.seqs.remove(seq)

    def exit(self):
        pass


def test_async_results_stream_and_cancel(monkeypatch):
    monkeypatch.setattr(llm_engine, 'LLMEngine', FakeEngine)
    async def run():
        async with AsyncLLMEngine('fake') as engine:
            a = engine.submit([1], SamplingParams(max_tokens=2), 'a')
            b = engine.submit([2], SamplingParams(max_tokens=20), 'b')
            b.cancel()
            assert [x async for x in a.stream()] == [42, 42]
            assert (await a.result())['request_id'] == 'a'
            try:
                await b.result()
                assert False
            except asyncio.CancelledError:
                pass
    asyncio.run(run())


def test_stream_receives_multiple_tokens_from_one_step(monkeypatch):
    class MultiTokenEngine(FakeEngine):
        def step(self):
            self.last_events = []
            for seq in self.seqs[:]:
                for i in range(2):
                    seq.append_token(42 + i)
                    seq.token_timestamps.append(1.0)
                    done = seq.num_completion_tokens == seq.max_tokens
                    self.last_events.append((seq, 42 + i, done))
                    if done:
                        self.seqs.remove(seq)
                        break
    monkeypatch.setattr(llm_engine, 'LLMEngine', MultiTokenEngine)
    async def run():
        async with AsyncLLMEngine('fake') as engine:
            handle = engine.submit([1], SamplingParams(max_tokens=2))
            assert [token async for token in handle.stream()] == [42, 43]
            assert (await handle.result())['token_ids'] == [42, 43]
    asyncio.run(run())
