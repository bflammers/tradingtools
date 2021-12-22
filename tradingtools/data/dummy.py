from decimal import Decimal

from typing import Dict
from random import uniform

import polars as pl

from ..utils import length_string_to_seconds
from .dataloader import AbstractData, AbstractDataLoader


class DummyData(AbstractData):
    def get_latest(self) -> Dict[str, Decimal]:
        return {pair: Decimal(uniform(0.1, 1000.0)) for pair in self.get_pairs()}

    def get_history(self, interval_length: str, bucket_length: str) -> pl.DataFrame:
        raise NotImplementedError


class DummyDataLoader(AbstractDataLoader):
    def data_factory(self) -> DummyData:
        return DummyData(self._config)
