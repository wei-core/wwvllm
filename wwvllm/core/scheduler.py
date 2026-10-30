from collections import deque
from time import monotonic

from wwvllm.core.sequence import SequenceStatus
from wwvllm.core.block_manager import BlockManager
from wwvllm.core.scheduler_output import ExecutionMode, ScheduledRequest, SchedulerOutput
from wwvllm.utils.exceptions import KVBlockExhaustedError


class Scheduler:
    def __init__(self, config):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.chunk_size = config.prefill_chunk_size
        self.eos = config.eos
        self.enable_speculative = getattr(config, "enable_speculative", False)
        self.speculative_num_tokens = getattr(config, "speculative_num_tokens", 0)
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
        while True:
            output = self._schedule_once()
            if output is not None:
                return output

    def _schedule_once(self):
        budget = self.max_num_batched_tokens
        scheduled = []
        # Decode is latency sensitive. Incomplete prefill requests rotate behind it.
        candidates = sorted(self.running, key=lambda s: s.num_cached_tokens < s.num_prompt_tokens)
        for seq in candidates:
            if budget == 0 or len(scheduled) >= self.max_num_seqs:
                break
            decode = seq.num_cached_tokens >= seq.num_prompt_tokens and seq.num_tokens - seq.num_cached_tokens == 1
            capacity = (len(seq.block_table) + len(self.block_manager.free_block_ids)) * self.block_size - seq.num_cached_tokens
            remaining_output = seq.max_tokens - seq.num_completion_tokens
            k = min(self.speculative_num_tokens, budget - 1, capacity - 1, remaining_output - 1)
            if decode and self.enable_speculative and seq.speculative_enabled and k > 0:
                transaction = self.block_manager.begin_transaction(seq, seq.num_tokens + k)
                if transaction is not None:
                    seq.num_scheduled_tokens = k + 1
                    scheduled.append(ScheduledRequest(seq, k + 1, ExecutionMode.SPECULATIVE, True, transaction))
                    budget -= k + 1
                    continue
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
                    raise KVBlockExhaustedError("sequence exceeds KV cache capacity")
                if len(self.running) == 1:
                    raise KVBlockExhaustedError("insufficient KV cache to make progress")
                self.preempt(victim)
                return None
            raise KVBlockExhaustedError("insufficient KV cache to schedule a request")
        # Round-robin requests within each phase.
        for item in scheduled:
            self.running.remove(item.seq)
            self.running.append(item.seq)
        return SchedulerOutput(scheduled, self.max_num_batched_tokens - budget)

    def postprocess(self, output, token_ids, speculative_results=None):
        tokens = iter(token_ids)
        events = []
        speculative_results = speculative_results or {}
        for item in output.scheduled:
            seq = item.seq
            if item.mode == ExecutionMode.SPECULATIVE:
                result = speculative_results[seq.seq_id]
                emitted = []
                for token in result.token_ids:
                    emitted.append(token)
                    if (not seq.ignore_eos and token == self.eos) or len(emitted) >= seq.max_tokens - seq.num_completion_tokens:
                        break
                accepted = min(result.accepted, len(emitted))
                # Append accepted tokens before hashing their completed KV blocks.
                for token in emitted[:accepted]:
                    seq.append_token(token)
                seq.num_scheduled_tokens = 1 + accepted
                self.block_manager.hash_blocks(seq)
                committed_end = seq.num_cached_tokens + seq.num_scheduled_tokens
                self.block_manager.finish_transaction(item.transaction, committed_end)
                seq.num_cached_tokens = committed_end
                seq.num_scheduled_tokens = 0
                for token in emitted[accepted:]:
                    seq.append_token(token)
                seq.draft_proposed += result.proposed
                seq.draft_accepted += accepted
                seq.target_verifications += 1
                for i, token in enumerate(emitted):
                    now = monotonic()
                    seq.token_timestamps.append(now)
                    finished = i == len(emitted) - 1 and ((not seq.ignore_eos and token == self.eos) or seq.num_completion_tokens >= seq.max_tokens)
                    if finished:
                        seq.status = SequenceStatus.FINISHED
                        seq.finish_time = now
                        self.block_manager.deallocate(seq)
                        self.running.remove(seq)
                    events.append((seq, token, finished))
                continue
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
