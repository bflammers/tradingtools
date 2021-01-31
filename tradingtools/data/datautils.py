import threading
import csv

import pandas as pd

from uuid import uuid4
from pathlib import Path
from time import sleep
from typing import Callable, Dict

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
]

# Create directory for results
def create_results_dir(parent_dir: Path):
    now = pd.Timestamp.now().strftime("%F_%T")
    ts_uuid = f"{now}_{uuid4().hex}"
    results_dir = (Path(parent_dir) / ts_uuid).absolute()
    results_dir.mkdir(parents=True, exist_ok=True)
    return results_dir


def make_ticker_callback(fn: Callable[[Dict], None]) -> Callable:
    async def ticker_callback(feed, pair, bid, ask, timestamp, receipt_timestamp):
        fn(
            {
                "feed": feed,
                "timestamp": timestamp,
                "receipt_timestamp": receipt_timestamp,
                "pair": pair,
                "bid": bid,
                "ask": ask,
            }
        )

    return ticker_callback


def make_trade_callback(fn: Callable[[Dict], None]) -> Callable:
    async def trade_callback(
        feed, pair, order_id, timestamp, side, amount, price, receipt_timestamp
    ):
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
            }
        )

    return trade_callback


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
    def __init__(self, path: Path, columns: list, single_row: bool = True) -> None:
        super().__init__(lifo=False)

        self.path = path
        self.columns = columns
        self._create_csv(self.path, self.columns)

        if single_row:
            self.add_consumer(consumer=self.append, interval_time=0)
        else:
            self.add_consumer(consumer=self.append_multiple, interval_time=0)

    @staticmethod
    def _create_csv(path: Path, columns: list) -> None:

        # Create parent dir if not exists
        path.parent.mkdir(parents=True, exist_ok=True)

        # Create file and write header
        with open(path, "w") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(columns)

    def append(self, new_values: dict) -> None:

        if new_values:

            row = []

            for column in self.columns:

                try:
                    row.append(new_values[column])
                except TypeError:
                    row.append(getattr(new_values, column))
                except KeyError:
                    row.append(None)
                    warnings.warn(
                        f"[CSVWriter.append] key-value pair for {column} not in new values for {self.path}"
                    )

            with open(self.path, "a") as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow(row)

    def append_multiple(
        self, new_values_list: list, add_uuid: bool = False, add_timestamp: bool = False
    ) -> None:

        # Common fields
        id = uuid4().hex
        timestamp = timestamp_to_string(pd.Timestamp.now())

        # Update volume for each symbol, add new if not yet present
        for new_values in new_values_list:

            if add_uuid:
                new_values["id"] = id

            if add_timestamp:
                new_values["timestamp"] = timestamp

            self.append(new_values=new_values)

    def read(self) -> pd.DataFrame:
        df = pd.read_csv(self.path)
        return df


if __name__ == "__main__":

    import numpy as np

    stream = ThreadStream()

    producer = lambda: {"a": np.random.randn()}
    consumer = lambda x: print(x, flush=True)

    stream.add_producer(producer, interval_time=0.1)
    stream.add_consumer(consumer, interval_time=0.5, batched=True)

    # TODO: batch not yet working!!!

    for i in range(100):
        # print(len(stream.get_latest()))
        sleep(0.1)

    exit()

    # CSVWriter

    p = Path("./data/testing/csv_writer.csv")
    csv_writer = CSVWriter(p, columns=["a", "b"])

    for i in range(100):
        csv_writer.add_to_q({"a": i, "b": i + 1})
        print(csv_writer.read())
