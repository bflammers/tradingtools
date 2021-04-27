import threading
import csv

import pandas as pd

from uuid import uuid4
from pathlib import Path
from time import sleep, time
from typing import Callable, Dict, List

try:
    from ..utils import warnings
    from ..utils import timestamp_to_string
    from .datahandling import ThreadStream
except ImportError:
    from tradingtools.utils import warnings
    from tradingtools.utils import timestamp_to_string
    from tradingtools.data.datahandling import ThreadStream

ticks_columns = [
    "feed",
    "timestamp",
    "receipt_timestamp",
    "pair",
    "bid",
    "ask",
    "write_time",
]

trades_columns = [
    "feed",
    "pair",
    "order_id",
    "timestamp",
    "side",
    "amount",
    "price",
    "receipt_timestamp",
    "write_time",
]

nbbo_columns = [
    "symbol",
    "bid",
    "bid_size",
    "ask",
    "ask_size",
    "bid_feed",
    "ask_feed",
    "write_time",
]

# Create directory for results
def create_results_dir(parent_dir: Path):
    now = pd.Timestamp.now().strftime("%F_%H%M%S")
    ts_uuid = f"{now}_{uuid4().hex[:5]}"
    results_dir = (Path(parent_dir) / ts_uuid).absolute()
    results_dir.mkdir(parents=True, exist_ok=True)
    return results_dir


def make_ticker_callback(fn: Callable[[Dict], None]) -> Callable:
    async def ticker_callback(feed, pair, bid, ask, timestamp, receipt_timestamp):
        write_time = time()
        fn(
            {
                "feed": feed,
                "timestamp": timestamp,
                "receipt_timestamp": receipt_timestamp,
                "pair": pair,
                "bid": bid,
                "ask": ask,
                "write_time": write_time,
            }
        )

    return ticker_callback


def make_trade_callback(fn: Callable[[Dict], None]) -> Callable:
    async def trade_callback(
        feed, pair, order_id, timestamp, side, amount, price, receipt_timestamp
    ):
        write_time = time()
        fn(
            {
                "feed": feed,
                "timestamp": timestamp,
                "receipt_timestamp": receipt_timestamp,
                "pair": pair,
                "order_id": order_id,
                "side": side,
                "amount": amount,
                "price": price,
                "write_time": write_time,
            }
        )

    return trade_callback


def make_nbbo_callback(fn: Callable[[Dict], None]) -> Callable:
    async def nbbo_callback(symbol, bid, bid_size, ask, ask_size, bid_feed, ask_feed):
        write_time = time()
        fn(
            {
                "symbol": symbol,
                "bid": bid,
                "bid_size": bid_size,
                "ask": ask,
                "ask_size": ask_size,
                "bid_feed": bid_feed,
                "ask_feed": ask_feed,
                "write_time": write_time,
            }
        )

    return nbbo_callback


def threadsafe_generator(f):
    """A decorator that takes a generator function and makes it thread-safe."""

    class ThreadSafeIterator:
        """Takes an iterator/generator and makes it thread-safe by
        serializing call to the `next` method of given iterator/generator.
        """

        def __init__(self, it):
            self.it = it
            self.lock = threading.Lock()

        def __iter__(self):
            return self

        def __next__(self):
            with self.lock:
                return self.it.__next__()

    def g(*a, **kw):
        return ThreadSafeIterator(f(*a, **kw))

    return g


class CSVWriter(ThreadStream):
    def __init__(self, path: Path, columns: list, nested: bool = False) -> None:
        super().__init__(lifo=False)

        self.path = path
        self.columns = columns
        self._create_csv(self.path, self.columns)

        if nested:
            self.add_consumer(
                consumer=self.append_nested, interval_time=0, batched=False
            )
        else:
            self.add_consumer(consumer=self.append, interval_time=0, batched=False)

    @staticmethod
    def _create_csv(path: Path, columns: list) -> None:

        # Create parent dir if not exists
        path.parent.mkdir(parents=True, exist_ok=True)

        # Create file and write header
        with open(path, "w") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(columns)

    def append(self, new_rows: List[dict]) -> None:

        if new_rows:

            with open(self.path, "a") as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=self.columns)
                writer.writerows(new_rows)

    def append_nested(
        self,
        new_rows_nested: List[List[dict]],
        add_uuid: bool = False,
        add_timestamp: bool = False,
    ) -> None:

        # Common fields
        id = uuid4().hex
        timestamp = timestamp_to_string(pd.Timestamp.now())

        # Update volume for each symbol, add new if not yet present
        for new_rows in new_rows_nested:

            if add_uuid or add_timestamp:

                for new_row in new_rows:

                    if add_uuid:
                        new_row["id"] = id

                    if add_timestamp:
                        new_row["timestamp"] = timestamp

            self.append(new_rows=new_rows)

    def read(self) -> pd.DataFrame:
        df = pd.read_csv(self.path)
        return df


if __name__ == "__main__":

    # import numpy as np

    # stream = ThreadStream()

    # producer = lambda: {"a": np.random.randn()}
    # consumer = lambda x: print(x, flush=True)

    # stream.add_producer(producer, interval_time=0.1)
    # stream.add_consumer(consumer, interval_time=0.5, batched=True)

    # # TODO: batch not yet working!!!

    # for i in range(100):
    #     # print(len(stream.get_latest()))
    #     sleep(0.1)

    # exit()

    # CSVWriter

    p = Path("./data/testing/csv_writer.csv")
    csv_writer = CSVWriter(p, columns=["a", "b"])

    for i in range(100):
        csv_writer.add_to_q({"a": i, "b": i + 1})
        print(csv_writer.read())
