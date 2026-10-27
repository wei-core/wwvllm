from collections import deque
from time import monotonic

from wwvllm.engine.sequence import SequenceStatus
from wwvllm.engine.block_manager import BlockManager
from wwvllm.engine.scheduler_output import ExecutionMode, ScheduledRequest, SchedulerOutput


class Scheduler:
    def __init__(self, config):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.chunk_size = config.prefill_chunk_size
        self.eos = config.eos
        self.block_size = config.kvcache_block_size
        self.block_manager = BlockManager(config.num_kvcache_blocks, self.block_size)
        self.waiting = deque()
        self.running = deque()

    def is_finished(self):
        return not self.waiting and not self.running

    def add(self, seq):
        if seq.num_tokens > len(self.block_manager.blocks) * self.block_size:
            raise ValueError("prompt exceeds KV cache capacity")
        self.waiting.append(seq)

    def cancel(self, seq):
        for queue in (self.waiting, self.running):
            if seq in queue:
                queue.remove(seq)
                self.block_manager.deallocate(seq)
                seq.status = SequenceStatus.CANCELLED
                seq.finish_time = monotonic()
                return True
        return False

    def preempt(self, seq):
        self.running.remove(seq)
        self.block_manager.deallocate(seq)
        seq.status = SequenceStatus.WAITING
        seq.preemption_count += 1
        self.waiting.appendleft(seq)

    def schedule(self):
        budget = self.max_num_batched_tokens
        scheduled = []
        # Decode is latency sensitive. Incomplete prefill requests rotate behind it.
        candidates = sorted(self.running, key=lambda s: s.num_cached_tokens < s.num_prompt_tokens)
        for seq in candidates:
            if budget == 0 or len(scheduled) >= self.max_num_seqs:
                break
            decode = seq.num_cached_tokens >= seq.num_prompt_tokens and seq.num_tokens - seq.num_cached_tokens == 1
            capacity = (len(seq.block_table) + len(self.block_manager.free_block_ids)) * self.block_size - seq.num_cached_tokens
            n = min(seq.num_tokens - seq.num_cached_tokens, budget, 1 if decode else self.chunk_size, capacity)
            if n > 0 and self.block_manager.reserve(seq, seq.num_cached_tokens + n):
                seq.num_scheduled_tokens = n
                scheduled.append(ScheduledRequest(seq, n, ExecutionMode.DECODE if decode else ExecutionMode.PREFILL, seq.num_cached_tokens + n == seq.num_tokens))
                budget -= n
        # Admit new requests only when there is an active sequence slot.
        attempts = len(self.waiting)
        while attempts and self.waiting and budget and len(self.running) < self.max_num_seqs and len(scheduled) < self.max_num_seqs:
            attempts -= 1
            seq = self.waiting.popleft()
            self.block_manager.attach_prefix(seq)
            capacity = (len(seq.block_table) + len(self.block_manager.free_block_ids)) * self.block_size - seq.num_cached_tokens
            n = min(seq.num_tokens - seq.num_cached_tokens, budget, self.chunk_size, capacity)
            if n <= 0 or not self.block_manager.reserve(seq, seq.num_cached_tokens + n):
                self.block_manager.deallocate(seq)
                self.waiting.append(seq)
                continue
            seq.status = SequenceStatus.RUNNING
            self.running.append(seq)
            seq.num_scheduled_tokens = n
            scheduled.append(ScheduledRequest(seq, n, ExecutionMode.PREFILL, seq.num_cached_tokens + n == seq.num_tokens))
            budget -= n
        if not scheduled:
            if self.running:
                # Recompute a victim later; fail explicitly if a single request cannot fit.
                victim = self.running[-1]
                if len(self.running) == 1 and victim.num_tokens > len(self.block_manager.blocks) * self.block_size:
                    raise RuntimeError("sequence exceeds KV cache capacity")
                if len(self.running) == 1:
                    raise RuntimeError("insufficient KV cache to make progress")
                self.preempt(victim)
                return self.schedule()
            raise RuntimeError("insufficient KV cache to schedule a request")
        # Round-robin requests within each phase.
        for item in scheduled:
            self.running.remove(item.seq)
            self.running.append(item.seq)
        return SchedulerOutput(scheduled, self.max_num_batched_tokens - budget)

    def postprocess(self, output, token_ids):
        tokens = iter(token_ids)
        events = []
        for item in output.scheduled:
            seq = item.seq
            self.block_manager.hash_blocks(seq)
            seq.num_cached_tokens += item.num_scheduled_tokens
            seq.num_scheduled_tokens = 0
            if not item.should_sample:
                continue
            token_id = next(tokens)
            seq.append_token(token_id)
            seq.token_timestamps.append(monotonic())
            finished = (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens >= seq.max_tokens
            if finished:
                seq.status = SequenceStatus.FINISHED
                seq.finish_time = monotonic()
                self.block_manager.deallocate(seq)
                self.running.remove(seq)
            events.append((seq, token_id, finished))
        return events
