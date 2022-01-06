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
    burn_in_interval: int = 0
    update_tol_interval: str = "5M"
    max_history_interval: str = "100D"
    hist__exchange: str = "Binance"
    hist__parent_path: str = "./data"
    hist__sleep_interval: str = None

    def _set_interval_seconds(
        self, attr_name: str, interval: str, falsy_value: int = 0
    ) -> None:

        if not interval:
            setattr(self, attr_name, falsy_value)
        else:
            interval_seconds = interval_to_seconds(interval)
            setattr(self, attr_name, interval_seconds)

    def __post_init__(self):

        self.interval_seconds = None
        self.burn_in_seconds = None
        self.update_tol_seconds = None
        self.sleep_seconds = None
        self.max_history_seconds = None

        self._set_interval_seconds("interval_seconds", self.interval)
        self._set_interval_seconds("burn_in_seconds", self.burn_in_interval)
        self._set_interval_seconds("update_tol_seconds", self.update_tol_interval)
        self._set_interval_seconds(
            "sleep_seconds", self.hist__sleep_interval, self.interval_seconds
        )
        self._set_interval_seconds(
            "max_history_seconds",
            self.max_history_interval,
            max(self.interval_seconds, self.update_tol_seconds),
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

    def get_complete(self) -> AbstractData:
        raise NotImplementedError

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
            data = self.data_factory()

            if data is None:
                return

            yield data

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
