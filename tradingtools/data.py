import os
import re
from pathlib import Path
from typing import Generator

import pandas as pd
import numpy as np


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

        for i, fn in enumerate(files):
            fp = Path(path) / Path(fn)
            print(f"[Dataloader] >> loading file {i}/{len(files)} >> {fp}")
            dfs.append(pd.read_csv(fp, skiprows=1).iloc[::-1])

        print(f"[Dataloader] >> concatenating files")
        self.df = pd.concat(dfs)

        # Modifying df
        self.df.columns = map(str.lower, self.df.columns)
        self.df["timestamp"] = pd.to_datetime(self.df["date"])
        self.df = self.df.set_index("timestamp", drop=False)
        self.df = self.df.drop(columns=["date", "unix timestamp"])

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

    p = "./data/cryptodatadownload/gemini/price"

    dl = HistoricalOHLCLoader("BTCUSD", p, "2019")
    dl.df.head()

    ticker = dl.get_ticker()

    for i, t in enumerate(ticker):
        print(t)
