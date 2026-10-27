from dataclasses import dataclass
from enum import Enum, auto


class ExecutionMode(Enum):
    PREFILL = auto()
    DECODE = auto()
    SPECULATIVE = auto()


@dataclass
class ScheduledRequest:
    seq: object
    num_scheduled_tokens: int
    mode: ExecutionMode
    should_sample: bool
    transaction: object = None


@dataclass
class SchedulerOutput:
    scheduled: list[ScheduledRequest]
    num_scheduled_tokens: int

    @property
    def has_prefill(self):
        return any(item.mode == ExecutionMode.PREFILL for item in self.scheduled)

    @property
    def has_decode(self):
        return any(item.mode in (ExecutionMode.DECODE, ExecutionMode.SPECULATIVE) for item in self.scheduled)
