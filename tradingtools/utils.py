from decimal import Decimal
import pandas as pd
import warnings
import threading
import csv

from uuid import uuid4
from pathlib import Path


def _warning_on_one_line(message, category, filename, lineno, file=None, line=None):
    return "%s:%s: %s: %s\n" % (filename, lineno, category.__name__, message)


warnings.formatwarning = _warning_on_one_line


class colors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


def color_number_sign(x: Decimal, decimals: int = 3, offset: float = 0) -> str:
    if (x - offset) > 0:
        return f"{colors.OKGREEN}+{x:.{decimals}f}{colors.ENDC}"
    else:
        return f"{colors.FAIL}{x:.{decimals}f}{colors.ENDC}"


def print_item(
    currency: str,
    value: Decimal,
    profit: Decimal = None,
    profit_percentage: Decimal = None,
    n_orders: int = None,
    n_open_orders: int = None,
) -> str:

    # value_colored = color_number_sign(value, decimals=2)
    out = f"{currency} {value:.3f}"

    if profit is not None:
        profit_colored = color_number_sign(profit)
        out += f" / {profit_colored} profit"

    if profit_percentage is not None:
        profit_percentage_colored = color_number_sign(profit_percentage)
        out += f" / {profit_percentage_colored} %"

    if n_orders is not None:
        out += f" / {n_orders} orders"

    if n_open_orders is not None:
        out += f" / {n_open_orders} open orders"

    return out


_strftime_format = "%Y-%m-%d %H:%M:%S.%f"


def timestamp_to_string(ts: pd.Timestamp) -> str:
    return ts.strftime(_strftime_format)


def string_to_timestamp(ts_string: str) -> pd.Timestamp:
    return pd.Timestamp(ts_string)


def extract_prices(tick: list, price_type: str = "close") -> dict:
    return {t["trading_pair"]: Decimal(t[price_type]) for t in tick if price_type in t}


class _threadsafe_iter:
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


def threadsafe_generator(f):
    """A decorator that takes a generator function and makes it thread-safe."""

    def g(*a, **kw):
        return _threadsafe_iter(f(*a, **kw))

    return g


class CSVWriter:
    def __init__(self, path: Path, columns: list) -> None:
        super().__init__()

        self.path = path
        self.columns = columns
        self._create_csv(self.path, self.columns)

    @staticmethod
    def _create_csv(path: str, columns: list) -> None:

        # Create file and write header
        with open(path, "w") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(columns)

    def append(self, new_values: dict) -> None:

        row = []

        for column in self.columns:

            try:
                row.append(new_values[column])
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