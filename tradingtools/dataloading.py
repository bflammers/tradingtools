import os
import re

from pathlib import Path
from typing import Generator
from decimal import Decimal

import ccxt

import pandas as pd
import numpy as np

try:
    from .utils import timestamp_to_string, threadsafe_generator
except:
    from utils import timestamp_to_string, threadsafe_generator


class DataLoader:
    def __init__(self, trading_pair: str, verbose: bool = False) -> None:
        super().__init__()

        self.df = None
        self.trading_pair = trading_pair
        self._verbose = verbose
        self._i = 0

    def get_single_tick(self) -> list:
        raise NotImplementedError("Base class")

    @threadsafe_generator
    def get_ticker(self) -> Generator:
        raise NotImplementedError("Base class")

    def get_df(self) -> pd.DataFrame:

        if self.df is None:
            Warning("DF not loaded")

        return self.df


class OHLCLoader(DataLoader):
    def __init__(
        self, trading_pair: str, exchange: str = "binance", verbose: bool = True
    ) -> None:
        super().__init__(trading_pair, verbose=verbose)
        self.trading_pair = trading_pair
        self.exchange_name = exchange
        self._tick_keys = ["timestamp", "open", "high", "low", "close", "volume"]
        self._latest_timestamp = None
        self._latest_close = None

        if self.exchange_name == "binance":
            self.exchange = ccxt.binance(
                {
                    "enableRateLimit": True,
                    "options": {
                        "adjustForTimeDifference": True,  # resolves the recvWindow timestamp error
                    },
                }
            )
        else:
            raise NotImplementedError("[OHLCLoader] currently only Binance supported")

    def get_single_tick(self) -> list:

        # Get data from exchange (single tick with limit=1, otherwise tick of last {limit} minutes)
        data = self.exchange.fetch_ohlcv(self.trading_pair, limit=1)

        # Transform to dict
        tick = {"trading_pair": self.trading_pair}
        for key, value in zip(self._tick_keys, data[-1]):

            if key in ["open", "high", "low", "close", "volume"]:
                tick[key] = Decimal(value)
            else:
                tick[key] = value

        self._latest_close = tick["close"]
        self._latest_timestamp = tick["timestamp"]

        return [tick]

    @threadsafe_generator
    def get_ticker(self) -> Generator:

        # Infinite generator
        while True:

            tick = self.get_single_tick()

            yield tick

    def __str__(self) -> str:

        ts = timestamp_to_string(pd.Timestamp(self._latest_timestamp, unit="ms"))
        out = f"[Dataloader] >> Tick timestamp: {ts} - close: {self._latest_close}"
        return out


class HistoricalOHLCLoader(DataLoader):
    def __init__(
        self,
        trading_pair: str,
        path: str,
        extra_pattern: str = None,
        verbose: bool = False,
    ) -> None:
        super().__init__(trading_pair, verbose)

        self._load_price_data(path, extra_pattern)

        self._i = 0
        self.n = self.df.shape[0]
        print(f"[Dataloader] >> shape loaded df: {self.df.shape}")

    def _load_price_data(self, path: str, extra_pattern: str) -> None:

        files = os.listdir(path)
        files = [fn for fn in files if re.findall(self.trading_pair, fn)]
        files.sort()

        if extra_pattern is not None:
            files = [fn for fn in files if re.findall(extra_pattern, fn)]

        dfs = []
        converters = {
            "open": Decimal,
            "high": Decimal,
            "low": Decimal,
            "close": Decimal,
            "volume": Decimal,
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
        self.df = self.df.rename({"symbol": "trading_pair"})
        self.df["timestamp"] = pd.to_datetime(self.df["date"])
        self.df = self.df.set_index("timestamp", drop=False)
        self.df = self.df.drop(columns=["date"])
        try:
            self.df = self.df.drop(columns=["unix timestamp"])
        except:
            self.df = self.df.drop(columns=["unix"])

    def get_single_tick(self) -> list:
        tick = self.df.iloc[self._i].to_dict()
        return [tick]

    @threadsafe_generator
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





if __name__ == "__main__":

    # import time

    # dl = OHLCLoader(symbol="BTC/EUR")

    # ticker = dl.get_ticker()

    # for tick in ticker:
    #     print(f"Tick: {tick}")
    #     time.sleep(5)

    # exit()

    p = "./data/cryptodatadownload/binance/price"

    dl = HistoricalOHLCLoader("BTCUSD", p, "dev")
    dl.df.head()

    ticker = dl.get_ticker()

    for i, t in enumerate(ticker):
        print(t)
        for _, x in t[0].items():
            print(type(x))
            print(x)
        exit()
