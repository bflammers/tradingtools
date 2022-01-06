from datetime import datetime
import unittest
import asyncio

import polars as pl
from typing import Dict
from decimal import Decimal
from pathlib import Path

from tradingtools.data import (
    DataLoaderConfig,
    DataLoaderTypes,
    dataloader_factory,
    HistoricalDataLoader,
    DataFrameData,
)

pairs = ["A/B", "C/D"]
dummy_data_path = Path(__file__).parent / "dummy_data"

default_config = DataLoaderConfig(
    type=DataLoaderTypes.historical,
    pairs=pairs,
    interval="1M",
    burn_in_interval=0,
    update_tol_interval="2M",
    hist__parent_path=dummy_data_path,
    hist__sleep_interval="0M",
)


class TestAsyncHistoricalDataLoader(unittest.IsolatedAsyncioTestCase):
    async def test_async_generator(self):

        dl = dataloader_factory(default_config)
        self.assertIsInstance(dl, HistoricalDataLoader)

        i = -1  # Start counting at -1 to match enumerate behavior
        async for data in dl.load_async():
            self.assertIsInstance(data, DataFrameData)
            i += 1

        # Data runs from 2021-09-04 01:00:00 to 2021-09-04 01:10:00, so 0 iterations
        self.assertEqual(i, 10)


class TestHistoricalDataLoader(unittest.TestCase):
    def test_generator(self):

        dl = dataloader_factory(default_config)
        self.assertIsInstance(dl, HistoricalDataLoader)

        for i, data in enumerate(dl.load()):
            self.assertIsInstance(data, DataFrameData)

        # Data runs from 2021-09-04 01:00:00 to 2021-09-04 01:10:00, so 10 iterations
        self.assertEqual(i, 10)

    def _test_get_latest_helper(
        self, prices: Dict[str, Decimal], ab_expected: Decimal, cd_expected: Decimal
    ) -> None:
        self.assertEqual(set(prices.keys()), set(pairs))
        self.assertEqual(prices["A/B"], ab_expected)
        self.assertEqual(prices["C/D"], cd_expected)

    def test_get_latest_vanilla(self):

        config = DataLoaderConfig(
            type=DataLoaderTypes.historical,
            pairs=pairs,
            interval="1M",
            burn_in_interval=0,
            update_tol_interval=0,
            hist__parent_path=dummy_data_path,
            hist__sleep_interval="0M",
        )

        dl = dataloader_factory(config)

        data = next(dl.load())
        self._test_get_latest_helper(data.get_latest(), Decimal("4.1"), None)

        data = next(dl.load())
        self._test_get_latest_helper(data.get_latest(), None, Decimal("40.1"))

    def test_get_latest_interval(self):

        config = DataLoaderConfig(
            type=DataLoaderTypes.historical,
            pairs=pairs,
            interval="7M",
            burn_in_interval=0,
            update_tol_interval=0,
            hist__parent_path=dummy_data_path,
            hist__sleep_interval="0M",
        )

        dl = dataloader_factory(config)

        data = next(dl.load())
        self._test_get_latest_helper(data.get_latest(), Decimal("4.1"), None)

        data = next(dl.load())
        self._test_get_latest_helper(data.get_latest(), Decimal("4.4"), Decimal("40.3"))

    def test_get_latest_update_tol(self):

        config = DataLoaderConfig(
            type=DataLoaderTypes.historical,
            pairs=pairs,
            interval="1M",
            burn_in_interval=0,
            update_tol_interval="2M",
            hist__parent_path=dummy_data_path,
            hist__sleep_interval="0M",
        )

        dl = dataloader_factory(config)

        data = next(dl.load())
        self._test_get_latest_helper(data.get_latest(), Decimal("4.1"), None)

        data = next(dl.load())
        self._test_get_latest_helper(data.get_latest(), Decimal("4.1"), Decimal("40.1"))

        data = next(dl.load())
        self._test_get_latest_helper(data.get_latest(), Decimal("4.2"), Decimal("40.1"))

        data = next(dl.load())
        self._test_get_latest_helper(data.get_latest(), Decimal("4.2"), None)

    def test_get_latest_burn_in(self):

        config = DataLoaderConfig(
            type=DataLoaderTypes.historical,
            pairs=pairs,
            interval="1M",
            burn_in_interval="1M",
            update_tol_interval=0,
            hist__parent_path=dummy_data_path,
            hist__sleep_interval="0M",
        )

        dl = dataloader_factory(config)

        data = next(dl.load())
        self._test_get_latest_helper(data.get_latest(), None, Decimal("40.1"))

        config = DataLoaderConfig(
            type=DataLoaderTypes.historical,
            pairs=pairs,
            interval="1M",
            burn_in_interval="5M",
            update_tol_interval=0,
            hist__parent_path=dummy_data_path,
            hist__sleep_interval="0M",
        )

        dl = dataloader_factory(config)

        data = next(dl.load())
        self._test_get_latest_helper(data.get_latest(), None, Decimal("40.2"))

    def test_get_history_vanilla(self):

        config = DataLoaderConfig(
            type=DataLoaderTypes.historical,
            pairs=pairs,
            interval="1M",
            burn_in_interval="3M",
            hist__parent_path=dummy_data_path,
            hist__sleep_interval="0M",
        )

        dl = dataloader_factory(config)
        data = next(dl.load())

        # Basics
        df = data.get_history(interval="1M", n_periods=3)
        self.assertIsInstance(df, pl.DataFrame)
        self.assertEqual(df.shape[0], 6)
        self.assertTrue(
            {"date", "symbol", "open", "low", "high", "close"}.issubset(set(df.columns))
        )

    def _test_get_history_helper(
        self,
        df: pl.DataFrame,
        last_datetime: datetime,
        first_datetime: datetime,
        n_rows: int,
        n_not_null_rows: int,
    ) -> None:
        self.assertEqual(df["date"].dt.to_python_datetime()[-1], last_datetime)
        self.assertEqual(df["date"].dt.to_python_datetime()[0], first_datetime)
        self.assertEqual(df.shape[0], n_rows)
        self.assertEqual(df.drop_nulls().shape[0], n_not_null_rows)

    def test_get_history_n_periods(self):

        config = DataLoaderConfig(
            type=DataLoaderTypes.historical,
            pairs=pairs,
            interval="1M",
            burn_in_interval="3M",
            hist__parent_path=dummy_data_path,
            hist__sleep_interval="0M",
        )

        dl = dataloader_factory(config)
        data = next(dl.load())

        df = data.get_history(interval="1M", n_periods=1)
        self._test_get_history_helper(
            df,
            last_datetime=datetime(2021, 9, 4, 1, 3),
            first_datetime=datetime(2021, 9, 4, 1, 3),
            n_rows=2,
            n_not_null_rows=0,
        )

        df = data.get_history(interval="1M", n_periods=2)
        self._test_get_history_helper(
            df,
            last_datetime=datetime(2021, 9, 4, 1, 3),
            first_datetime=datetime(2021, 9, 4, 1, 2),
            n_rows=4,
            n_not_null_rows=1,
        )

        df = data.get_history(interval="1M", n_periods=3)
        self._test_get_history_helper(
            df,
            last_datetime=datetime(2021, 9, 4, 1, 3),
            first_datetime=datetime(2021, 9, 4, 1, 1),
            n_rows=6,
            n_not_null_rows=2,
        )

        df = data.get_history(interval="1M", n_periods=4)
        self._test_get_history_helper(
            df,
            last_datetime=datetime(2021, 9, 4, 1, 3),
            first_datetime=datetime(2021, 9, 4, 1, 0),
            n_rows=8,
            n_not_null_rows=3,
        )

        df = data.get_history(interval="1M", n_periods=5)
        self._test_get_history_helper(
            df,
            last_datetime=datetime(2021, 9, 4, 1, 3),
            first_datetime=datetime(2021, 9, 4, 1, 0),
            n_rows=8,
            n_not_null_rows=3,
        )

    def test_get_history_interval(self):
        
        config = DataLoaderConfig(
            type=DataLoaderTypes.historical,
            pairs=pairs,
            interval="1M",
            burn_in_interval="5M",
            hist__parent_path=dummy_data_path,
            hist__sleep_interval="0M",
        )

        dl = dataloader_factory(config)
        data = next(dl.load())

        df = data.get_history(interval="1M", n_periods=None)
        self._test_get_history_helper(
            df,
            last_datetime=datetime(2021, 9, 4, 1, 5),
            first_datetime=datetime(2021, 9, 4, 1, 0),
            n_rows=12,
            n_not_null_rows=4,
        )
        self.assertEqual(df["close"][0], 4.1)
        self.assertEqual(df["close"][-1], 40.2)

        df = data.get_history(interval="2M", n_periods=None)
        self._test_get_history_helper(
            df,
            last_datetime=datetime(2021, 9, 4, 1, 4),
            first_datetime=datetime(2021, 9, 4, 1, 0),
            n_rows=6,
            n_not_null_rows=4,
        )
        self.assertEqual(df["close"][0], 4.1)
        self.assertEqual(df["close"][-1], 40.2)

        df = data.get_history(interval="2M", n_periods=None)
        self._test_get_history_helper(
            df,
            last_datetime=datetime(2021, 9, 4, 1, 4),
            first_datetime=datetime(2021, 9, 4, 1, 0),
            n_rows=6,
            n_not_null_rows=4,
        )
        self.assertEqual(df["close"][0], 4.1)
        self.assertEqual(df["close"][-1], 40.2)

        df = data.get_history(interval="3M", n_periods=None)
        self._test_get_history_helper(
            df,
            last_datetime=datetime(2021, 9, 4, 1, 3),
            first_datetime=datetime(2021, 9, 4, 1, 0),
            n_rows=4,
            n_not_null_rows=3,
        )
        self.assertEqual(df["close"][0], 4.2)
        self.assertEqual(df["close"][-1], 40.2)
        
        df = data.get_history(interval="4M", n_periods=None)
        self._test_get_history_helper(
            df,
            last_datetime=datetime(2021, 9, 4, 1, 4),
            first_datetime=datetime(2021, 9, 4, 1, 0),
            n_rows=4,
            n_not_null_rows=3,
        )
        self.assertEqual(df["close"][0], 4.2)
        self.assertEqual(df["close"][-1], 40.2)


if __name__ == "__main__":

    unittest.main()
