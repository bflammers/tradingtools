import threading

import pandas as pd

from time import sleep
from queue import Queue, LifoQueue, Empty
from typing import Dict, List, Tuple

try:
    from ..utils import warnings
    from ..utils import timestamp_to_string
except ImportError:
    from tradingtools.utils import warnings
    from tradingtools.utils import timestamp_to_string


class ThreadStream:
    q: Queue
    stop_event: threading.Event = threading.Event()
    latest: list = []
    latest_when_empty: bool = False
    block_until_next: bool = False
    debugging: bool = False
    threads: List[threading.Thread] = []

    def __init__(
        self,
        lifo: bool = True,
        latest_when_empty: bool = False,
        max_q_size: int = 100000,
        debugging: bool = False
    ) -> None:

        self.debugging = debugging
        self.latest_when_empty = latest_when_empty

        if lifo:
            self.q = LifoQueue(maxsize=max_q_size)
        else:
            self.q = Queue(maxsize=max_q_size)

    def add_consumer(self, consumer, interval_time: int = 0, batched: bool = False):

        if batched and self.latest_when_empty:
            warnings.warn(
                "[Threadstream.add_consumer] latest_when_empty=True has no effect with batched=True"
            )

        kwargs = {
            "worker": self._consumer_worker,
            "fn": consumer,
            "interval_time": interval_time,
            "batched": batched,
        }

        self.start_worker(**kwargs)

    def add_producer(self, producer, interval_time: int = 0):

        if self.latest_when_empty:
            self.latest = producer()
            
        self.start_worker(
            worker=self._producer_worker, fn=producer, interval_time=interval_time
        )

    def _producer_worker(self, fn, interval_time) -> None:

        while not self.stop_event.is_set():
            x = fn()
            self.q.put(x)
            sleep(interval_time)

        print("[ThreadStream] Producer worker closed", flush=True)

    def _consumer_worker(self, fn, interval_time, batched) -> None:

        if interval_time > 0 or batched:
            # Interval time > 0 --> sleep takes care of timing of next iter
            # Batched --> empty queue must not block
            self.block_until_next = False
        else:
            self.block_until_next = True

        if batched:
            getter = self.empty_q
        else:
            getter = self.get_next

        while not self.stop_event.is_set():
            
            if self.debugging:
                q_size = self.q.qsize()
                if q_size > 0:
                    print(f"Q size: {q_size}")
                    
            x = getter()
            fn(x)
            sleep(interval_time)

        print("[ThreadStream] Consumer worker closed", flush=True)

    def start_worker(self, worker, **kwargs):
        thr = threading.Thread(target=worker, kwargs=kwargs)
        thr.daemon = True
        thr.start()
        self.threads.append(thr)

    def terminate(self):
        self.stop_event.set()

    def add_to_q(self, item):
        self.q.put(item=item, block=False)

    def get_next(self):

        try:
            self.latest = self.q.get(block=self.block_until_next)
            self.q.task_done()
        except Empty:
            if not self.latest_when_empty:
                return []

        return [self.latest]

    def empty_q(self):

        fresh = True
        empty = False
        records = []

        while not empty:

            try:
                x = self.q.get(block=False)
                records.append(x)
                self.q.task_done()

            except Empty:
                if fresh:
                    x = self.q.get(block=True)
                    records.append(x)
                    self.q.task_done()
                else:
                    empty = True

            fresh = False

        return records

    def __exit__(self):
        self.stop_event.set()
        print("[ThreadStream] exit -- worker thread closed", flush=True)


class MultiStream(ThreadStream):
    streams: Dict[str, Tuple[ThreadStream, bool]] = dict()

    def __init__(self, interval_time: int) -> None:
        super().__init__()
        self.add_producer(producer=self._multi_producer, interval_time=interval_time)

    def _multi_producer(self):

        stream_items = dict()

        for name, (stream, batched) in self.streams.items():

            if batched:
                stream_items[name] = stream.empty_q()
            else:
                stream_items[name] = stream.get_next()

        return stream_items

    def add_producer_stream(self, name, stream: ThreadStream, batched: bool = True):
        self.streams[name] = (stream, batched)

    def add_callback(self, callback):
        self.start_worker(self._callback_worker, fn=callback)


if __name__ == "__main__":

    import numpy as np
    import time

    start_time = time.time()

    def cb(tick):
        now = time.time()
        print(now - start_time, tick, flush=True)

    ms = MultiStream(1)

    s1 = ThreadStream(latest_when_empty=True)
    s1.add_producer(lambda: {"s1": np.random.randn()}, interval_time=0.5)
    ms.add_producer_stream("s1", s1, batched=False)

    s2 = ThreadStream(latest_when_empty=True)
    s2.add_producer(lambda: {"s2": np.random.randn()}, interval_time=2)
    ms.add_producer_stream("s2", s2, batched=False)

    ms.add_callback(callback=cb)

    for i in range(100):
        sleep(0.1)
