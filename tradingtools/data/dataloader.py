import asyncio
from decimal import Decimal
import time

from dataclasses import dataclass
from enum import Enum

from typing import AsyncIterator, List, Dict

import polars as pl

from ..utils import length_string_to_seconds


class DataLoaderTypes(Enum):
    dummy = "dummy"


@dataclass
class DataLoaderConfig:
    type: DataLoaderTypes
    pairs: List[str]
    interval_length: str

    @property
    def interval_seconds(self) -> int:

        # When False, empty string or zero return zero
        if not self.interval_length:
            return 0

        return length_string_to_seconds(self.interval_length)


class AbstractData:

    def __init__(self, config: DataLoaderConfig) -> None:
        self._config: DataLoaderConfig = config

    def get_pairs(self) -> List[str]:
        return self._config.pairs

    def get_latest(self) -> Dict[str, Decimal]:
        raise NotImplementedError

    def get_history(self, interval_length: str, bucket_length: str) -> pl.DataFrame:
        raise NotImplementedError


class AbstractDataLoader:

    def __init__(self, config: DataLoaderConfig) -> None:
        self._config: DataLoaderConfig = config

    def get_pairs(self) -> List[str]:
        return self._config.pairs

    def data_factory(self) -> AbstractData:
        raise NotImplementedError

    async def load(self) -> AsyncIterator[AbstractData]:

        while True:

            # Starting time - before yielding data and control to eventloop
            t_start = time.perf_counter()

            # Yield data object
            yield self.data_factory()

            # Determine sleep time and sleep
            t_diff = time.perf_counter() - t_start
            t_sleep = self._config.interval_seconds - t_diff
            await asyncio.sleep(t_sleep)
