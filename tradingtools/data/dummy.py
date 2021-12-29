from decimal import Decimal

from typing import Dict
from random import uniform

import polars as pl

from ..utils import length_string_to_seconds
from .dataloader import AbstractData, AbstractDataLoader, DataLoaderConfig


class DummyData(AbstractData):
    _prices: dict

    def __init__(self, config: DataLoaderConfig) -> None:
        super().__init__(config)
        self._prices = {} 

    def get_latest(self) -> Dict[str, Decimal]:
        return self._prices

    def get_history(self, interval_length: str, bucket_length: str) -> pl.DataFrame:
        raise NotImplementedError


class DummyDataLoader(AbstractDataLoader):

    def __init__(self, config: DataLoaderConfig) -> None:
        super().__init__(config)
        self._prices = {pair: Decimal(uniform(0.1, 300.0)) for pair in self.get_pairs()}

    def data_factory(self) -> DummyData:
        data = DummyData(self._config)

        for pair, price in self._prices.items():
            self._prices[pair] = price * Decimal(uniform(0.95, 1.05))

        data._prices = self._prices
        return data