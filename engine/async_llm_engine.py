"""Single-owner worker for asynchronous, streaming inference requests."""
import asyncio
from concurrent.futures import Future
from dataclasses import dataclass
from queue import Queue, Empty
from threading import Thread
from time import monotonic
from uuid import uuid4

from wwvllm.sampling_params import SamplingParams

_END = object()


@dataclass
class RequestHandle:
    request_id: str
    result_future: asyncio.Future
    output_queue: asyncio.Queue
    _engine: object
    arrival_time: float

    async def result(self):
        return await self.result_future

    def cancel(self):
        self._engine.cancel(self.request_id)

    async def stream(self):
        while True:
            item = await self.output_queue.get()
            if item is _END:
                break
            yield item
        # Propagate worker errors and cancellation to stream consumers.
        await self.result_future


class AsyncLLMEngine:
    def __init__(self, model, **kwargs):
        self.loop = asyncio.get_running_loop()
        self.commands = Queue()
        self.handles = {}
        self.limit = kwargs.get("max_pending_requests", 1024)
        self.closed = False
        self.ready = Future()
        self.worker = Thread(target=self._run, args=(model, kwargs), daemon=True)
        self.worker.start()

    async def start(self):
        await asyncio.wrap_future(self.ready)
        return self

    def submit(self, prompt, sampling_params=None, request_id=None):
        if self.closed:
            raise RuntimeError("engine is closed")
        if len(self.handles) >= self.limit:
            raise RuntimeError("request queue is full")
        request_id = str(request_id) if request_id is not None else uuid4().hex
        if request_id in self.handles:
            raise ValueError("duplicate request_id")
        handle = RequestHandle(request_id, self.loop.create_future(), asyncio.Queue(), self, monotonic())
        self.handles[request_id] = handle
        handle.result_future.add_done_callback(lambda f: self.cancel(request_id) if f.cancelled() else None)
        self.commands.put(("add", handle, prompt, sampling_params or SamplingParams()))
        return handle

    def cancel(self, request_id):
        if request_id in self.handles and not self.closed:
            self.commands.put(("cancel", request_id))

    def _deliver(self, handle, token=None, result=None, error=None, cancelled=False):
        if handle.request_id not in self.handles:
            return
        if token is not None:
            handle.output_queue.put_nowait(token)
        if result is not None or error is not None or cancelled:
            self.handles.pop(handle.request_id, None)
            handle.output_queue.put_nowait(_END)
            if not handle.result_future.done():
                if cancelled:
                    handle.result_future.cancel()
                elif error is not None:
                    handle.result_future.set_exception(error)
                else:
                    handle.result_future.set_result(result)

    def _run(self, model, kwargs):
        engine = None
        active = {}
        try:
            from wwvllm.engine.llm_engine import LLMEngine
            engine = LLMEngine(model, **kwargs)
            self.ready.set_result(None)
            stopping = False
            while not stopping:
                try:
                    command = self.commands.get(block=not active, timeout=None if not active else 0)
                except Empty:
                    command = None
                while command:
                    op, *args = command
                    if op == "add":
                        handle, prompt, params = args
                        try:
                            seq = engine.add_request(prompt, params)
                            seq.request_id = handle.request_id
                            seq.arrival_time = handle.arrival_time
                            active[handle.request_id] = (seq, handle)
                        except Exception as exc:
                            self.loop.call_soon_threadsafe(self._deliver, handle, None, None, exc)
                    elif op == "cancel":
                        pair = active.pop(args[0], None)
                        if pair:
                            engine.scheduler.cancel(pair[0])
                            self.loop.call_soon_threadsafe(self._deliver, pair[1], None, None, None, True)
                    else:
                        stopping = True
                        break
                    try:
                        command = self.commands.get_nowait()
                    except Empty:
                        command = None
                if active and not stopping:
                    engine.step()
                    for seq, token, finished in engine.last_events:
                        handle = active[seq.request_id][1]
                        result = None
                        if finished:
                            active.pop(seq.request_id)
                            result = {"request_id": seq.request_id, "token_ids": seq.completion_token_ids,
                                      "text": engine.tokenizer.decode(seq.completion_token_ids),
                                      "arrival_time": seq.arrival_time, "token_timestamps": seq.token_timestamps,
                                      "finish_time": seq.finish_time, "preemption_count": seq.preemption_count}
                        self.loop.call_soon_threadsafe(self._deliver, handle, token, result)
        except BaseException as exc:
            if not self.ready.done():
                self.ready.set_exception(exc)
            self.loop.call_soon_threadsafe(self._fail_all, exc)
        finally:
            if engine is not None:
                try:
                    engine.exit()
                except Exception:
                    pass
            self.loop.call_soon_threadsafe(self._fail_all, RuntimeError("engine stopped"))

    def _fail_all(self, error):
        self.closed = True
        for handle in list(self.handles.values()):
            self._deliver(handle, error=error)

    async def close(self):
        if not self.closed:
            self.closed = True
            self.commands.put(("stop",))
        await asyncio.to_thread(self.worker.join)

    async def __aenter__(self):
        return await self.start()

    async def __aexit__(self, *args):
        await self.close()
