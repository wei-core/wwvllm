from copy import copy
from enum import Enum, auto
from itertools import count
from time import monotonic

from wwvllm.sampling_params import SamplingParams


class SequenceStatus(Enum):
    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()
    CANCELLED = auto()
    FAILED = auto()


class Sequence:
    block_size = 256
    counter = count()

    def __init__(self, token_ids: list[int], sampling_params = SamplingParams()):
        if not token_ids:
            raise ValueError("prompt must contain at least one token")
        self.seq_id = next(Sequence.counter)
        self.request_id = str(self.seq_id)
        self.arrival_time = monotonic()
        self.token_timestamps = []
        self.finish_time = None
        self.preemption_count = 0
        self.status = SequenceStatus.WAITING
        self.token_ids = copy(token_ids)
        self.last_token = token_ids[-1]
        self.num_tokens = len(self.token_ids)
        self.num_prompt_tokens = len(token_ids)
        self.num_cached_tokens = 0
        self.num_scheduled_tokens = 0
        self.is_prefill = True
        self.block_table = []
        self.temperature = sampling_params.temperature
        self.max_tokens = sampling_params.max_tokens
        self.ignore_eos = sampling_params.ignore_eos
        self.top_k = sampling_params.top_k
        self.top_p = sampling_params.top_p
        self.speculative_enabled = sampling_params.speculative
        self.draft_proposed = 0
        self.draft_accepted = 0
        self.target_verifications = 0

    def __len__(self):
        return self.num_tokens

    def __getitem__(self, key):
        return self.token_ids[key]

    @property
    def is_finished(self):
        return self.status in (SequenceStatus.FINISHED, SequenceStatus.CANCELLED, SequenceStatus.FAILED)

    @property
    def num_completion_tokens(self):
        return self.num_tokens - self.num_prompt_tokens

    @property
    def prompt_token_ids(self):
        return self.token_ids[:self.num_prompt_tokens]

    @property
    def completion_token_ids(self):
        return self.token_ids[self.num_prompt_tokens:]

    @property
    def num_blocks(self):
        return (self.num_tokens + self.block_size - 1) // self.block_size

    @property
    def last_block_num_tokens(self):
        return self.num_tokens - (self.num_blocks - 1) * self.block_size

    def block(self, i):
        assert 0 <= i < self.num_blocks
        return self.token_ids[i*self.block_size: (i+1)*self.block_size]

    def append_token(self, token_id: int):
        self.token_ids.append(token_id)
        self.last_token = token_id
        self.num_tokens += 1

    def __getstate__(self):
        # Workers need arbitrary token slices after chunking or recomputation.
        return self.__dict__

    def __setstate__(self, state):
        self.__dict__.update(state)
