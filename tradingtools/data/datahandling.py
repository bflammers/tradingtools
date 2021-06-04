import threading
import logging
import atexit

import multiprocessing as mp
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

logger = logging.getLogger(__name__)


class Stream:
    block_until_next: bool = False
    debugging: bool = False
    workers: List = []
    q = None
    stop_event = None

    def __init__(self, debugging: bool = False) -> None:

        self.debugging = debugging
        atexit.register(self.terminate)

    def add_consumer(self, consumer, interval_time: int = 0, batched: bool = False):

        self.start_worker(
            worker=self._consumer_worker,
            fn=consumer,
            interval_time=interval_time,
            batched=batched,
        )

    def add_producer(self, producer, interval_time: int = 0):

        self.start_worker(
            worker=self._producer_worker, fn=producer, interval_time=interval_time
        )

    def _producer_worker(self, fn, interval_time) -> None:

        logger.info("Producer worker starting")

        while not self.stop_event.is_set():
            x = fn()
            self.q.put(x)
            sleep(interval_time)

        logger.info("Producer worker closed")

    def _consumer_worker(self, fn, interval_time, batched) -> None:

        if interval_time > 0 or batched:
            # Interval time > 0 --> sleep takes care of timing of next iter
            # Batched --> empty queue must not block
            self.block_until_next = False
        else:
            # Items are processed the moment they come in
            self.block_until_next = True

        if batched:
            getter = self.empty_q
        else:
            getter = self.get_next

        logger.info("Consumer worker starting")

        while not self.stop_event.is_set():

            if self.debugging:
                q_size = self.q.qsize()
                if q_size > 0:
                    logger.debug(f"Q size: {q_size}")

            x = getter()
            fn(x)
            sleep(interval_time)

        logger.info("Consumer worker closed")

    def start_worker(self, worker, **kwargs):
        raise NotImplementedError

    def terminate(self):
        self.stop_event.set()

        for worker in self.workers:
            
            try:
                worker.terminate()
            except AttributeError:
                pass 

            worker.join()

    def add_to_q(self, item):
        self.q.put(item, block=False)

    def get_next(self, raise_empty=False, block_override=False):
        raise NotImplementedError

    def empty_q(self):

        fresh = True
        empty = False
        records = []

        while not empty:
            try:
                x = self.get_next(raise_empty=True)
                records.append(x)

            except Empty:
                if fresh:
                    x = self.get_next(block_override=True)
                    records.append(x)

                else:
                    empty = True

            fresh = False

        return records


class ProcessStream(Stream):
    # m: mp.Manager
    q: mp.Queue
    stop_event: mp.Event
    workers: List[mp.Process] = []

    def __init__(
        self,
        max_q_size: int = 100000,
        debugging: bool = False,
    ) -> None:

        super().__init__(debugging)

        # self.m = mp.Manager()
        self.q = mp.Queue(maxsize=max_q_size)
        self.stop_event = mp.Event()

    def start_worker(self, worker, **kwargs):
        proc = mp.Process(target=worker, kwargs=kwargs)
        # proc.daemon = True
        proc.start()
        self.workers.append(proc)

    def get_next(self, raise_empty=False, block_override=False):

        try:
            return self.q.get(block=block_override or self.block_until_next)

        except Empty as e:

            if raise_empty:
                raise e

            # Return none if Empty exception should not be raised
            return None


class ThreadStream(Stream):
    q: Queue
    stop_event: threading.Event = threading.Event()
    workers: List[threading.Thread] = []

    def __init__(self, max_q_size: int = 100000, debugging: bool = False) -> None:
        super().__init__(debugging)
        self.q = Queue(maxsize=max_q_size)

    def start_worker(self, worker, **kwargs):
        thr = threading.Thread(target=worker, kwargs=kwargs)
        thr.daemon = True
        thr.start()
        self.workers.append(thr)

    def get_next(self, raise_empty=False, block_override=False):

        try:
            item = self.q.get(block=block_override or self.block_until_next)
            self.q.task_done()
            return item

        except Empty as e:

            if raise_empty:
                raise e

            # Return None if Empty exception should not be raised
            return None


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
