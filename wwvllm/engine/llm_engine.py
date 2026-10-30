import atexit
from dataclasses import fields
from time import perf_counter
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import torch.multiprocessing as mp

from wwvllm.config import Config
from wwvllm.sampling_params import SamplingParams
from wwvllm.core.sequence import Sequence
from wwvllm.core.scheduler import Scheduler
from wwvllm.executor.worker import ModelRunner
from wwvllm.core.scheduler_output import ExecutionMode


class LLMEngine:

    def __init__(self, model, **kwargs):
        config_fields = {field.name for field in fields(Config)}
        unknown = kwargs.keys() - config_fields
        if unknown:
            raise TypeError(f"Unknown engine options: {', '.join(sorted(unknown))}")
        config = Config(model, **kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        if config.enable_speculative:
            from transformers import AutoConfig
            draft_tokenizer = AutoTokenizer.from_pretrained(config.speculative_model, use_fast=True)
            draft_config = AutoConfig.from_pretrained(config.speculative_model)
            if (self.tokenizer.get_vocab() != draft_tokenizer.get_vocab()
                    or self.tokenizer.all_special_ids != draft_tokenizer.all_special_ids
                    or config.hf_config.vocab_size != draft_config.vocab_size):
                raise ValueError("draft and target tokenizers and vocabularies must match")
            if draft_config.model_type != "qwen3":
                raise ValueError("draft model must be a Qwen3 causal language model")
            if config.max_model_len > draft_config.max_position_embeddings:
                raise ValueError("draft model context is shorter than max_model_len")
        Sequence.block_size = config.kvcache_block_size
        self.config = config
        self._closed = False
        self.ps = []
        self.events = []
        ctx = mp.get_context("spawn")
        for i in range(1, config.tensor_parallel_size):
            event = ctx.Event()
            process = ctx.Process(target=ModelRunner, args=(config, i, event))
            process.start()
            self.ps.append(process)
            self.events.append(event)
        self.model_runner = ModelRunner(config, 0, self.events)
        config.eos = self.tokenizer.eos_token_id
        self.scheduler = Scheduler(config)
        atexit.register(self.exit)

    def exit(self):
        if self._closed:
            return
        self._closed = True
        self.model_runner.call("exit")
        del self.model_runner
        for p in self.ps:
            p.join()

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        if len(prompt) + sampling_params.max_tokens > self.config.max_model_len:
            raise ValueError("prompt and output exceed max_model_len")
        seq = Sequence(prompt, sampling_params)
        self.scheduler.add(seq)
        return seq

    def step(self):
        output = self.scheduler.schedule()
        normal = [item for item in output.scheduled if item.mode != ExecutionMode.SPECULATIVE]
        speculative = [item for item in output.scheduled if item.mode == ExecutionMode.SPECULATIVE]
        try:
            token_ids = []
            if normal:
                seqs = [item.seq for item in normal]
                flags = [item.should_sample for item in normal]
                has_prefill = any(item.mode == ExecutionMode.PREFILL for item in normal)
                token_ids = self.model_runner.call("run", seqs, has_prefill, flags)
            results = {item.seq.seq_id: self.model_runner.call("run_speculative", item.seq, item.num_scheduled_tokens - 1)
                       for item in speculative}
            self.last_events = self.scheduler.postprocess(output, token_ids, results)
        except BaseException:
            for item in speculative:
                if item.transaction.active:
                    self.scheduler.block_manager.rollback_transaction(item.transaction)
                    item.seq.num_scheduled_tokens = 0
            raise
        outputs = [(seq.seq_id, seq.completion_token_ids) for seq, _, finished in self.last_events if finished]
        num_tokens = output.num_scheduled_tokens if output.has_prefill else -output.num_scheduled_tokens
        return outputs, num_tokens

    def is_finished(self):
        return self.scheduler.is_finished()

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[str]:
        pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True, disable=not use_tqdm)
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        if len(prompts) != len(sampling_params):
            raise ValueError("one sampling parameter set is required per prompt")
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)
        outputs = {}
        prefill_throughput = decode_throughput = 0.
        while not self.is_finished():
            t = perf_counter()
            output, num_tokens = self.step()
            if num_tokens > 0:
                prefill_throughput = num_tokens / (perf_counter() - t)
            else:
                decode_throughput = -num_tokens / (perf_counter() - t)
            pbar.set_postfix({
                "Prefill": f"{int(prefill_throughput)}tok/s",
                "Decode": f"{int(decode_throughput)}tok/s",
            })
            for seq_id, token_ids in output:
                outputs[seq_id] = token_ids
                pbar.update(1)
        pbar.close()
        outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]
        outputs = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids} for token_ids in outputs]
        return outputs
