import pickle
from copy import copy
from collections import OrderedDict
import torch
import torch.distributed as dist
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory

from wwvllm.config import Config
from wwvllm.core.sequence import Sequence
from wwvllm.models.registry import get_model_class
from wwvllm.layers.sampler import Sampler
from wwvllm.utils.context import set_context, get_context, reset_context
from wwvllm.models.loader import load_model
from wwvllm.layers.speculative_sampler import probabilities, sample, verify


class ModelRunner:

    def __init__(self, config: Config, rank: int, event: Event | list[Event]):
        self.config = config
        hf_config = config.hf_config
        self.block_size = config.kvcache_block_size
        self.enforce_eager = config.enforce_eager or getattr(config.hf_config, "model_type", None) == "qwen3_moe"
        self.world_size = config.tensor_parallel_size
        self.rank = rank
        self.enable_mixed_graph = config.enable_mixed_graph
        self.mixed_graphs = OrderedDict()
        self.mixed_graph_stats = {"captures": 0, "replays": 0, "evictions": 0}
        self.event = event

        dist.init_process_group("nccl", "tcp://localhost:2333", world_size=self.world_size, rank=rank)
        torch.cuda.set_device(rank)
        default_dtype = torch.get_default_dtype()
        try:
            torch.set_default_dtype(hf_config.dtype)
            torch.set_default_device("cuda")
            self.model = get_model_class(hf_config)(hf_config)
            if config.quantization in ('awq', 'gptq'):
                from wwvllm.quantization.checkpoint import load_quantized_model
                load_quantized_model(self.model, config.model, hf_config.quantization_config)
            else:
                load_model(self.model, config.model)
                if config.quantization:
                    from wwvllm.quantization.linear import quantize_model
                    quantize_model(self.model, 4 if config.quantization == 'int4' else 8, config.quant_group_size)
            self.sampler = Sampler()
            self.draft_runner = None
            if config.enable_speculative:
                from wwvllm.executor.draft_runner import DraftRunner
                torch.set_default_device("cpu")
                try:
                    self.draft_runner = DraftRunner(config.speculative_model, hf_config.dtype)
                finally:
                    torch.set_default_device("cuda")
            self.warmup_model()
            self.allocate_kv_cache()
            if not self.enforce_eager:
                self.capture_cudagraph()
        except BaseException:
            dist.destroy_process_group()
            raise
        finally:
            torch.set_default_device("cpu")
            torch.set_default_dtype(default_dtype)

        if self.world_size > 1:
            if rank == 0:
                self.shm = SharedMemory(name="nanovllm", create=True, size=2**20)
                dist.barrier()
            else:
                dist.barrier()
                self.shm = SharedMemory(name="nanovllm")
                self.loop()

    def exit(self):
        if self.world_size > 1:
            self.shm.close()
            dist.barrier()
            if self.rank == 0:
                self.shm.unlink()
        if not self.enforce_eager:
            del self.graphs, self.graph_pool, self.graph_vars
        torch.cuda.synchronize()
        self.mixed_graphs.clear()
        del self.model, self.kv_cache, self.draft_runner
        torch.cuda.empty_cache()
        dist.destroy_process_group()

    def loop(self):
        while True:
            method_name, args = self.read_shm()
            self.call(method_name, *args)
            if method_name == "exit":
                break

    def read_shm(self):
        assert self.world_size > 1 and self.rank > 0
        self.event.wait()
        n = int.from_bytes(self.shm.buf[0:4], "little")
        method_name, *args = pickle.loads(self.shm.buf[4:n+4])
        self.event.clear()
        return method_name, args

    def write_shm(self, method_name, *args):
        assert self.world_size > 1 and self.rank == 0
        data = pickle.dumps([method_name, *args])
        n = len(data)
        self.shm.buf[0:4] = n.to_bytes(4, "little")
        self.shm.buf[4:n+4] = data
        for event in self.event:
            event.set()

    def call(self, method_name, *args):
        if self.world_size > 1 and self.rank == 0:
            self.write_shm(method_name, *args)
        method = getattr(self, method_name, None)
        return method(*args)

    def warmup_model(self):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        max_num_batched_tokens, max_model_len = self.config.max_num_batched_tokens, self.config.max_model_len
        seq_len = min(max_num_batched_tokens, max_model_len)
        num_seqs = min(max_num_batched_tokens // seq_len, self.config.max_num_seqs)
        seqs = [Sequence([0] * seq_len) for _ in range(num_seqs)]
        for seq in seqs:
            seq.num_scheduled_tokens = seq_len
        self.run(seqs, True)
        torch.cuda.empty_cache()

    def allocate_kv_cache(self):
        config = self.config
        hf_config = config.hf_config
        free, total = torch.cuda.mem_get_info()
        used = total - free
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]
        num_kv_heads = hf_config.num_key_value_heads // self.world_size
        head_dim = getattr(hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads)
        block_bytes = 2 * hf_config.num_hidden_layers * self.block_size * num_kv_heads * head_dim * hf_config.dtype.itemsize
        draft_reserve = int(total * config.draft_memory_reserve) if config.enable_speculative else 0
        config.num_kvcache_blocks = int(total * config.gpu_memory_utilization - used - peak + current - draft_reserve) // block_bytes
        if config.num_kvcache_blocks <= 0:
            raise RuntimeError("insufficient GPU memory for target KV cache and draft reservation")
        self.kv_cache = torch.empty(2, hf_config.num_hidden_layers, config.num_kvcache_blocks, self.block_size, num_kv_heads, head_dim)
        layer_id = 0
        for module in self.model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                module.k_cache = self.kv_cache[0, layer_id]
                module.v_cache = self.kv_cache[1, layer_id]
                layer_id += 1

    def prepare_block_tables(self, seqs: list[Sequence]):
        max_len = max(len(seq.block_table) for seq in seqs)
        block_tables = [seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs]
        block_tables = torch.tensor(block_tables, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        return block_tables

    def prepare_prefill(self, seqs: list[Sequence], sample_flags=None):
        input_ids = []
        positions = []
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        max_seqlen_q = 0
        max_seqlen_k = 0
        slot_mapping = []
        block_tables = None
        for seq in seqs:
            start = seq.num_cached_tokens
            seqlen_q = seq.num_scheduled_tokens
            end = start + seqlen_q
            seqlen_k = end
            input_ids.extend(seq[start:end])
            positions.extend(range(start, end))
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)
            if not seq.block_table:    # warmup
                continue
            start_block = start // self.block_size
            end_block = (end + self.block_size - 1) // self.block_size
            for i in range(start_block, end_block):
                slot_start = seq.block_table[i] * self.block_size
                if i == start_block:
                    slot_start += start % self.block_size
                if i != end_block - 1:
                    slot_end = seq.block_table[i] * self.block_size + self.block_size
                else:
                    slot_end = seq.block_table[i] * self.block_size + end - i * self.block_size
                slot_mapping.extend(range(slot_start, slot_end))
        if cu_seqlens_k[-1] > cu_seqlens_q[-1]:    # prefix cache
            block_tables = self.prepare_block_tables(seqs)
        sample_indices_list = [cu_seqlens_q[i + 1] - 1 for i, flag in enumerate(sample_flags or [True] * len(seqs)) if flag]
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        if sample_flags is None:
            sample_flags = [True] * len(seqs)
        sample_indices = torch.tensor(sample_indices_list, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        set_context(True, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, None, block_tables, sample_indices)
        return input_ids, positions

    def prepare_decode(self, seqs: list[Sequence]):
        input_ids = []
        positions = []
        slot_mapping = []
        context_lens = []
        for seq in seqs:
            input_ids.append(seq.last_token)
            positions.append(len(seq) - 1)
            context_lens.append(len(seq))
            slot_mapping.append(seq.block_table[-1] * self.block_size + seq.last_block_num_tokens  - 1)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        context_lens = torch.tensor(context_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        block_tables = self.prepare_block_tables(seqs)
        set_context(False, slot_mapping=slot_mapping, context_lens=context_lens, block_tables=block_tables)
        return input_ids, positions

    def prepare_sample(self, seqs: list[Sequence]):
        temperatures = [seq.temperature for seq in seqs]
        temperatures = torch.tensor(temperatures, dtype=torch.float32, pin_memory=True).cuda(non_blocking=True)
        return temperatures

    @torch.inference_mode()
    def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill: bool):
        if is_prefill or self.enforce_eager or input_ids.size(0) > 512:
            return self.model.compute_logits(self.model(input_ids, positions))
        else:
            bs = input_ids.size(0)
            context = get_context()
            graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]
            graph_vars = self.graph_vars
            graph_vars["input_ids"][:bs] = input_ids
            graph_vars["positions"][:bs] = positions
            graph_vars["slot_mapping"].fill_(-1)
            graph_vars["slot_mapping"][:bs] = context.slot_mapping
            graph_vars["context_lens"].zero_()
            graph_vars["context_lens"][:bs] = context.context_lens
            graph_vars["block_tables"][:bs, :context.block_tables.size(1)] = context.block_tables
            graph.replay()
            return self.model.compute_logits(graph_vars["outputs"][:bs])

    @torch.inference_mode()
    def run_mixed_graph(self, input_ids, positions):
        context = get_context()
        assert context.is_prefill and context.block_tables is not None
        bs = context.cu_seqlens_q.numel() - 1
        # Capture exact metadata shapes and host-side attention bounds.
        key = (input_ids.numel(), bs, context.max_seqlen_q,
               context.max_seqlen_k, context.block_tables.size(1))
        cached = self.mixed_graphs.get(key)
        if cached is None:
            ids = torch.empty_like(input_ids)
            pos = torch.empty_like(positions)
            qlens = torch.empty_like(context.cu_seqlens_q)
            klens = torch.empty_like(context.cu_seqlens_k)
            slots = torch.empty_like(context.slot_mapping)
            tables = torch.empty_like(context.block_tables)
            cached = dict(ids=ids, pos=pos, qlens=qlens, klens=klens, slots=slots, tables=tables)
        for name, value in [('ids', input_ids), ('pos', positions),
                            ('qlens', context.cu_seqlens_q), ('klens', context.cu_seqlens_k),
                            ('slots', context.slot_mapping), ('tables', context.block_tables)]:
            cached[name].copy_(value)
        if 'graph' not in cached:
            # Each warmup/capture uses the real input and writes the same KV
            # values as replay; no candidate or sampling state is advanced.
            set_context(True, cached['qlens'], cached['klens'], context.max_seqlen_q,
                        context.max_seqlen_k, cached['slots'], None, cached['tables'])
            try:
                self.model(cached['ids'], cached['pos'])
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, self.graph_pool):
                    outputs = self.model(cached['ids'], cached['pos'])
                cached['graph'] = graph
                cached['outputs'] = outputs
            finally:
                set_context(context.is_prefill, context.cu_seqlens_q, context.cu_seqlens_k,
                            context.max_seqlen_q, context.max_seqlen_k, context.slot_mapping,
                            context.context_lens, context.block_tables, context.sample_indices)
            self.mixed_graph_stats['captures'] += 1
            self.mixed_graphs[key] = cached
            if len(self.mixed_graphs) > self.config.max_mixed_graphs:
                self.mixed_graphs.popitem(last=False)
                self.mixed_graph_stats['evictions'] += 1
        else:
            self.mixed_graphs.move_to_end(key)
        cached['graph'].replay()
        self.mixed_graph_stats['replays'] += 1
        return self.model.compute_logits(cached['outputs'])

    def run(self, seqs: list[Sequence], is_prefill: bool, sample_flags=None) -> list[int]:
        sample_flags = sample_flags if sample_flags is not None else [True] * len(seqs)
        input_ids, positions = self.prepare_prefill(seqs, sample_flags) if is_prefill else self.prepare_decode(seqs)
        temperatures = self.prepare_sample([seq for seq, flag in zip(seqs, sample_flags) if flag]) if self.rank == 0 else None
        try:
            mixed = False
            if (self.enable_mixed_graph and self.world_size == 1 and not self.config.quantization
                    and getattr(self.config.hf_config, 'model_type', None) == 'qwen3'
                    and not self.enforce_eager and is_prefill):
                decode_flags = [s.num_cached_tokens >= s.num_prompt_tokens and
                                s.num_tokens - s.num_cached_tokens == 1 and s.num_scheduled_tokens == 1
                                for s in seqs]
                mixed = any(decode_flags) and not all(decode_flags)
            logits = (self.run_mixed_graph(input_ids, positions) if mixed
                      else self.run_model(input_ids, positions, is_prefill))
            if self.rank == 0 and any(sample_flags):
                sampled_seqs = [seq for seq, flag in zip(seqs, sample_flags) if flag]
                if any(seq.top_k or seq.top_p < 1 for seq in sampled_seqs):
                    token_ids = []
                    for row, seq in zip(logits, sampled_seqs):
                        probs = probabilities(row, seq.temperature, seq.top_k, seq.top_p)
                        token_ids.append(int(probs.argmax()) if seq.temperature == 0 else sample(probs))
                else:
                    token_ids = self.sampler(logits, temperatures).tolist()
            else:
                token_ids = []
            return token_ids
        finally:
            reset_context()

    @torch.inference_mode()
    def run_speculative(self, seq, count):
        if self.draft_runner is None:
            raise RuntimeError("draft model is not loaded")
        draft_ids, draft_probs = self.draft_runner.propose(seq, count)
        verify_seq = copy(seq)
        verify_seq.token_ids = seq.token_ids + draft_ids
        verify_seq.num_tokens = len(verify_seq.token_ids)
        verify_seq.num_scheduled_tokens = count + 1
        input_ids, positions = self.prepare_prefill([verify_seq], [True])
        try:
            get_context().sample_indices = torch.arange(count + 1, device="cuda")
            logits = self.run_model(input_ids, positions, True)
            target_probs = [probabilities(row, seq.temperature, seq.top_k, seq.top_p) for row in logits]
            return verify(draft_ids, draft_probs, target_probs, greedy=seq.temperature == 0)
        finally:
            reset_context()

    @torch.inference_mode()
    def capture_cudagraph(self):
        config = self.config
        hf_config = config.hf_config
        max_bs = min(self.config.max_num_seqs, 512)
        max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size
        input_ids = torch.zeros(max_bs, dtype=torch.int64)
        positions = torch.zeros(max_bs, dtype=torch.int64)
        slot_mapping = torch.zeros(max_bs, dtype=torch.int32)
        context_lens = torch.zeros(max_bs, dtype=torch.int32)
        block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)
        outputs = torch.zeros(max_bs, hf_config.hidden_size)
        self.graph_bs = sorted(set([x for x in [1, 2, 4, 8] + list(range(16, max_bs + 1, 16)) if x <= max_bs] + [max_bs]))
        self.graphs = {}
        self.graph_pool = None

        for bs in reversed(self.graph_bs):
            graph = torch.cuda.CUDAGraph()
            set_context(False, slot_mapping=slot_mapping[:bs], context_lens=context_lens[:bs], block_tables=block_tables[:bs])
            outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # warmup
            with torch.cuda.graph(graph, self.graph_pool):
                outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # capture
            if self.graph_pool is None:
                self.graph_pool = graph.pool()
            self.graphs[bs] = graph
            torch.cuda.synchronize()
            reset_context()

        self.graph_vars = dict(
            input_ids=input_ids,
            positions=positions,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            outputs=outputs,
        )
