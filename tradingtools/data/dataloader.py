import asyncio
from datetime import datetime
from decimal import Decimal
import time

from dataclasses import dataclass, field
from enum import Enum

from typing import AsyncIterator, Iterator, List, Dict

import polars as pl

from ..utils import interval_to_seconds


class DataLoaderTypes(Enum):
    dummy = "dummy"
    historical = "historical"


@dataclass
class DataLoaderConfig:
    type: DataLoaderTypes
    pairs: List[str]
    interval: str = "1M"
    hist__exchange: str = "Binance"
    hist__parent_path: str = "./data"
    hist__burn_in_periods: int = 10
    hist__update_tol_interval: str = "5M"
    hist__sleep_interval: str = None
    hist__max_history_interval: str = "100D"

    def __post_init__(self):

        # interval_length to interval_seconds
        if self.interval is None:
            self.interval_seconds = 0
        else:
            self.interval_seconds = interval_to_seconds(self.interval)

        # burn in periods to burin_in_seconds
        if not self.hist__burn_in_periods:
            self.burn_in_seconds = 0
        else:
            self.burn_in_seconds = self.interval_seconds * self.hist__burn_in_periods

        # historical__update_tol_length to historical__update_tol_seconds
        if self.hist__update_tol_interval is None:
            self.hist__update_tol_seconds = 0
        else:
            self.hist__update_tol_seconds = interval_to_seconds(
                self.hist__update_tol_interval
            )

        # sleep interval (overrides interval if set) to seconds
        if self.hist__sleep_interval is None:
            self.sleep_seconds = self.interval_seconds
        else:
            self.sleep_seconds = interval_to_seconds(self.hist__sleep_interval)

        # historical__history_limit_length to historical__history_limit_seconds
        if self.hist__max_history_interval is None:
            self.max_history_seconds = interval_to_seconds("1000D")
        else:
            self.max_history_seconds = interval_to_seconds(
                self.hist__max_history_interval
            )


class AbstractData:
    def __init__(self, config: DataLoaderConfig) -> None:
        self._config: DataLoaderConfig = config

    def get_pairs(self) -> List[str]:
        return self._config.pairs

    def get_latest(self) -> Dict[str, Decimal]:
        raise NotImplementedError

    def get_history(self, interval: str, n_periods: int) -> pl.DataFrame:
        raise NotImplementedError


class AbstractDataLoader:
    def __init__(self, config: DataLoaderConfig) -> None:
        self._config: DataLoaderConfig = config

    def get_pairs(self) -> List[str]:
        return self._config.pairs

    def data_factory(self) -> AbstractData:
        raise NotImplementedError

    def _sleep_time(self, start_time: float) -> float:

        t_diff = time.perf_counter() - start_time
        t_sleep = self._config.sleep_seconds - t_diff
        return max(0, t_sleep)

    async def load_async(self) -> AsyncIterator[AbstractData]:

        while True:

            # Starting time - before yielding data and control to eventloop
            t_start = time.perf_counter()

            # Yield data object
            yield self.data_factory()

            # Determine sleep time and sleep
            t_sleep = self._sleep_time(t_start)
            await asyncio.sleep(t_sleep)

    def load(self) -> Iterator[AbstractData]:

        while True:

            # Starting time - before yielding data and control to eventloop
            t_start = time.perf_counter()

            # Yield data object
            data = self.data_factory()

            if data is None:
                return

            yield data

            # Determine sleep time and sleep
            t_sleep = self._sleep_time(t_start)
            time.sleep(t_sleep)
