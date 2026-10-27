from collections import deque
import xxhash
import numpy as np

from wwvllm.engine.sequence import Sequence


class Block:

    def __init__(self, block_id):
        self.block_id = block_id
        self.ref_count = 0
        self.hash = -1
        self.token_ids = []

    def update(self, hash: int, token_ids: list[int]):
        self.hash = hash
        self.token_ids = token_ids

    def reset(self):
        self.ref_count = 1
        self.hash = -1
        self.token_ids = []


class BlockManager:

    def __init__(self, num_blocks: int, block_size: int):
        self.block_size = block_size
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]
        self.hash_to_block_id: dict[int, int] = dict()
        self.free_block_ids: deque[int] = deque(range(num_blocks))
        self.used_block_ids: set[int] = set()

    @classmethod
    def compute_hash(cls, token_ids: list[int], prefix: int = -1):
        h = xxhash.xxh64()
        if prefix != -1:
            h.update(prefix.to_bytes(8, "little"))
        h.update(np.array(token_ids).tobytes())
        return h.intdigest()

    def _allocate_block(self) -> int:
        block_id = self.free_block_ids.popleft()
        block = self.blocks[block_id]
        assert block.ref_count == 0
        if block.hash != -1 and self.hash_to_block_id.get(block.hash) == block_id:
            del self.hash_to_block_id[block.hash]
        block.reset()
        self.used_block_ids.add(block_id)
        return block_id

    def _deallocate_block(self, block_id: int):
        assert self.blocks[block_id].ref_count == 0
        self.used_block_ids.remove(block_id)
        self.free_block_ids.append(block_id)

    def attach_prefix(self, seq: Sequence):
        """Attach only complete cached blocks; leave a token to compute for sampling."""
        h = -1
        for i in range((seq.num_tokens - 1) // self.block_size):
            tokens = seq.block(i)
            h = self.compute_hash(tokens, h)
            block_id = self.hash_to_block_id.get(h)
            if block_id is None or self.blocks[block_id].token_ids != tokens:
                break
            block = self.blocks[block_id]
            if block.ref_count == 0:
                self.free_block_ids.remove(block_id)
                self.used_block_ids.add(block_id)
            block.ref_count += 1
            seq.block_table.append(block_id)
            seq.num_cached_tokens += self.block_size

    def reserve(self, seq: Sequence, end: int) -> bool:
        needed = max(0, (end + self.block_size - 1) // self.block_size - len(seq.block_table))
        if needed > len(self.free_block_ids):
            return False
        for _ in range(needed):
            seq.block_table.append(self._allocate_block())
        return True

    def deallocate(self, seq: Sequence):
        for block_id in reversed(seq.block_table):
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self._deallocate_block(block_id)
        seq.num_cached_tokens = 0
        seq.block_table.clear()

    def hash_blocks(self, seq: Sequence):
        start = seq.num_cached_tokens // self.block_size
        end = (seq.num_cached_tokens + seq.num_scheduled_tokens) // self.block_size
        if start == end: return
        h = self.blocks[seq.block_table[start - 1]].hash if start > 0 else -1
        for i in range(start, end):
            block = self.blocks[seq.block_table[i]]
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h)
            block.update(h, token_ids)
            self.hash_to_block_id[h] = block.block_id

    def begin_transaction(self, seq: Sequence, end: int):
        """Reserve speculative slots without changing the committed KV cursor."""
        if getattr(seq, '_kv_transaction', None) is not None:
            raise RuntimeError('sequence already has an active KV transaction')
        old_blocks = len(seq.block_table)
        if end < seq.num_cached_tokens or not self.reserve(seq, end):
            return None
        transaction = KVTransaction(seq, seq.num_cached_tokens, old_blocks, end)
        seq._kv_transaction = transaction
        return transaction

    def finish_transaction(self, transaction, committed_end: int):
        seq = transaction.seq
        if not transaction.active or getattr(seq, '_kv_transaction', None) is not transaction:
            raise RuntimeError('inactive KV transaction')
        if not transaction.old_cached_len <= committed_end <= transaction.reserved_end:
            raise ValueError('commit outside reserved KV range')
        keep = max(transaction.old_blocks, (committed_end + self.block_size - 1) // self.block_size)
        for block_id in reversed(seq.block_table[keep:]):
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self._deallocate_block(block_id)
        del seq.block_table[keep:]
        transaction.active = False
        seq._kv_transaction = None

    def rollback_transaction(self, transaction):
        self.finish_transaction(transaction, transaction.old_cached_len)


class KVTransaction:
    def __init__(self, seq, old_cached_len, old_blocks, reserved_end):
        self.seq = seq
        self.old_cached_len = old_cached_len
        self.old_blocks = old_blocks
        self.reserved_end = reserved_end
        self.active = True
