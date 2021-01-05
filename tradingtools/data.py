import os
import re
from pathlib import Path
from typing import Generator
from decimal import Decimal

import ccxt

import pandas as pd
import numpy as np

try:
    from .utils import timestamp_to_string
except:
    from utils import timestamp_to_string


class DataLoader:
    def __init__(self, symbol: str, verbose: bool = False) -> None:
        super().__init__()

        self.df = None
        self.symbol = symbol
        self._verbose = verbose
        self._i = 0

    def get_ticker(self) -> Generator:
        raise NotImplementedError("Base class")

    def get_df(self) -> pd.DataFrame:

        if self.df is None:
            Warning("DF not loaded")

        return self.df


class OHLCLoader(DataLoader):
    def __init__(
        self, symbol: str, exchange: str = "binance", verbose: bool = True
    ) -> None:
        super().__init__(symbol, verbose=verbose)
        self.symbol = symbol
        self.exchange_name = exchange
        self._tick_keys = ["timestamp", "open", "high", "low", "close", "volume"]
        self._latest_timestamp = None
        self._latest_close = None

        if self.exchange_name == "binance":
            self.exchange = ccxt.binance()
        else:
            raise NotImplementedError("[OHLCLoader] currently only Binance supported")

    def get_ticker(self) -> Generator:

        # Infinite generator
        while True:

            # Get data from exchange (single tick with limit=1, otherwise tick of last {limit} minutes)
            data = self.exchange.fetch_ohlcv(self.symbol, limit=1)

            # Transform to dict
            tick = {"symbol": self.symbol}
            for key, value in zip(self._tick_keys, data[-1]):

                if key in ['open', 'high', 'low', 'close', 'volume']:
                    tick[key] = Decimal(value)
                else:
                    tick[key] = value

            self._latest_close = tick['close']
            self._latest_timestamp = tick['timestamp']

            yield [tick]

    def __str__(self) -> str:

        ts = timestamp_to_string(
            pd.Timestamp(self._latest_timestamp, unit="ms")
        )
        out = f"[Dataloader] >> Tick timestamp: {ts} - close: {self._latest_close}"
        return out


class HistoricalOHLCLoader(DataLoader):
    def __init__(
        self, symbol: str, path: str, extra_pattern: str = None, verbose: bool = False
    ) -> None:
        super().__init__(symbol, verbose)

        self._load_price_data(path, extra_pattern)

        self.n = self.df.shape[0]
        print(f"[Dataloader] >> shape loaded df: {self.df.shape}")

    def _load_price_data(self, path: str, extra_pattern: str) -> None:

        files = os.listdir(path)
        files = [fn for fn in files if re.findall(self.symbol, fn)]
        files.sort()

        if extra_pattern is not None:
            files = [fn for fn in files if re.findall(extra_pattern, fn)]

        dfs = []
        converters = {
            "Open": Decimal,
            "High": Decimal,
            "Low": Decimal,
            "Close": Decimal,
            "Volume": Decimal,
        }

        for i, fn in enumerate(files):
            fp = Path(path) / Path(fn)
            print(f"[Dataloader] >> loading file {i}/{len(files)} >> {fp}")
            dfs.append(pd.read_csv(fp, skiprows=1, converters=converters).iloc[::-1])

        print(f"[Dataloader] >> concatenating files")
        self.df = pd.concat(dfs)

        # Modifying df
        self.df.columns = map(str.lower, self.df.columns)
        self.df["timestamp"] = pd.to_datetime(self.df["date"])
        self.df = self.df.set_index("timestamp", drop=False)
        self.df = self.df.drop(columns=["date"])
        try:
            self.df = self.df.drop(columns=["unix timestamp"])
        except:
            self.df = self.df.drop(columns=["unix"])

    def get_ticker(self) -> Generator:
        for self._i, (index, row) in enumerate(self.df.iterrows()):

            if self._verbose:
                print(f"[Dataloader] >> processing row {self._i}/{self.n} >> {index}")

            yield [row.to_dict()]

    def _resample_df(self, freq: str = "5T") -> None:
        self.df = self.df.resample(freq, label="right", closed="right").agg(
            {
                "close": ["median", "std", "last"],
                "low": "min",
                "high": "max",
                "volume": "sum",
            }
        )
        self.df.columns = self.df.columns.map("_".join)
        self.df = self.df.rename(
            columns={
                "close_last": "close",
                "low_min": "low",
                "high_max": "high",
                "volume_sum": "volume",
            }
        )

    def __str__(self) -> str:
        row = self.df.iloc[self._i]
        return f"[Dataloader] >> processing row {self._i}/{self.n} >> {row.name}"


class CombinationLoader(DataLoader):
    def __init__(self, symbol: str) -> None:
        super().__init__(symbol)
        self.dataloaders = dict()
        raise NotImplementedError

    def add_dataloader(self, key: str, dataloader: DataLoader) -> None:
        self.dataloaders[key] = dataloader

    def get_ticker(self) -> Generator:
        """Generator that provides an iterator over all dataloaders"""
        raise NotImplementedError()


if __name__ == "__main__":

    import time

    dl = OHLCLoader(symbol="BTC/EUR")

    ticker = dl.get_ticker()

    for tick in ticker:
        print(f"Tick: {tick}")
        time.sleep(5)

    exit()

    p = "./data/cryptodatadownload/gemini/price"

    dl = HistoricalOHLCLoader("BTCUSD", p, "2019")
    dl.df.head()

    ticker = dl.get_ticker()

    for i, t in enumerate(ticker):
        print(t)
        for _, x in t[0].items():
            print(type(x))
            print(x)
        exit()
