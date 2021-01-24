import threading

from time import sleep
from queue import Queue, Empty
from typing import Dict


class TickHandler:
    def __init__(self, callback) -> None:
        self.callback = callback
        self.q = Queue()
        self.stop_event = threading.Event()

    def process(self, tick):
        self.q.put(tick)

    def _worker(self) -> None:

        while not self.stop_event.is_set():
            x = self.q.get()
            self.callback(x)
            self.q.task_done()

        print("[TickHandler._worker] Worker closed", flush=True)

    def start_worker(self):
        thr = threading.Thread(target=self._worker)
        thr.daemon = True
        thr.start()
        self.worker_running = True

    def __exit__(self):
        self.stop_event.set()
        print("[TickHandler] exit -- worker thread closed", flush=True)


class BatchedTickHandler(TickHandler):
    def __init__(self, callback, interval_seconds) -> None:
        self.interval_seconds = interval_seconds
        super().__init__(callback)

    def _worker(self) -> None:

        while not self.stop_event.is_set():
            records = self._empty_q()
            self.callback(records)
            sleep(self.interval_seconds)

        print("[TickHandler._worker] Worker closed", flush=True)

    def _empty_q(self):

        empty = False
        records = []
        while not empty:

            try:
                x = self.q.get(block=False)
                records.append(x)
                self.q.task_done()
            except Empty:
                empty = True

        return records


class CombinedHandler(BatchedTickHandler):
    handlers: Dict[str, BatchedTickHandler] = dict()

    def add_handler(self, name):
        self.handlers[name] = BatchedTickHandler(None, None)
        return self.handlers[name]

    def _worker(self) -> None:

        while not self.stop_event.is_set():

            handler_records = dict()

            for name, handler in self.handlers.items():
                handler_records[name] = handler._empty_q()
            
            self.callback(handler_records)
            sleep(self.interval_seconds)

        print("[TickHandler._worker] Worker closed", flush=True)
    

if __name__ == "__main__":

    def cb(tick):
        print(tick, flush=True)

    # th = BatchedTickHandler(cb, 1)
    # th.start_worker()

    # for i in range(100):
    #     th.process(i)
    #     sleep(0.2)

    # print("----")
    # print(th._empty_q())

    th = CombinedHandler(cb, 0.14)

    a = th.add_handler("a")
    b = th.add_handler("b")

    th.start_worker()

    for i in range(100):
        a.process(i)
        b.process(i + 100)
        sleep(0.1)

    print("----")
    print(a._empty_q())
    print(b._empty_q())
