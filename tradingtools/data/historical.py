import re
import polars as pl

from logging import currentframe, getLogger
from decimal import Decimal
from typing import Dict, List, Tuple
from random import uniform
from pathlib import Path
from datetime import datetime, timedelta

from ..utils import (
    float_to_decimal,
    interval_to_seconds,
    split_pair,
)
from .dataloader import AbstractData, AbstractDataLoader, DataLoaderConfig
from .data_utils import prepare_skeleton, to_polars_duration, filter_period


logger = getLogger(__name__)


class DataFrameData(AbstractData):
    def __init__(self, config: DataLoaderConfig) -> None:
        super().__init__(config)
        self._df = None
        # ASSUMES A 1M GRAIN
        self._n_rows_update_tol = max(1, self._config.update_tol_seconds // 60) * len(
            self.get_pairs()
        )

    def get_pairs(self) -> List[str]:
        return self._config.pairs

    def set_up(self, df: pl.DataFrame) -> None:
        self._df = df

    def get_latest(self) -> Dict[str, Decimal]:

        # Select last known closing prices
        df_latest = (
            self._df.tail(self._n_rows_update_tol)
            .groupby("symbol", maintain_order=True)
            .agg(pl.col("close").drop_nulls().last())
        )

        # Collect prices, convert prices to decimal, collect
        latest_prices = df_latest.to_dict(as_series=False)
        prices = dict(
            zip(
                latest_prices["symbol"],
                latest_prices["close_last"],
            )
        )
        latest = {pair: float_to_decimal(prices.get(pair)) for pair in self.get_pairs()}

        return latest

    def _aggregate(self, df: pl.DataFrame, interval: str) -> pl.DataFrame:

        pl_duration = to_polars_duration(interval)
        df = (
            df.with_column(pl.col("date").dt.truncate(pl_duration))
            .groupby(["date", "symbol"], maintain_order=True)
            .agg(
                [
                    pl.col("open").drop_nulls().first().alias("open"),
                    pl.col("high").drop_nulls().max().alias("high"),
                    pl.col("low").drop_nulls().min().alias("low"),
                    pl.col("close").drop_nulls().last().alias("close"),
                    pl.col("Volume USDT").sum().alias("Volume USDT"),
                    pl.col("tradecount").sum().alias("tradecount"),
                ]
            )
        )
        return df

    def get_history(self, interval: str = "1M", n_periods: int = None) -> pl.DataFrame:

        df = self._df.clone().drop(name="unix")

        # Initial filter to keep interval * n_periods of records
        # we do this initial rough filter to not aggregate an
        # unnecessary large number records
        if n_periods is not None:
            n_seconds = interval_to_seconds(interval) * n_periods
            df = filter_period(df, length_seconds=n_seconds)

        # Check if aggregation is needed
        if interval != "1M":
            df = self._aggregate(df, interval=interval)

        if n_periods is None:
            return df

        n_datetimes = df["date"].n_unique()
        if n_datetimes < n_periods:
            logger.warning(
                f"[get_history] n_periods is {n_periods} but data only has {n_datetimes} unique periods"
            )
        else:
            n_records = len(self.get_pairs()) * n_periods
            df = df.tail(n_records)

        return df


class HistoricalDataLoader(AbstractDataLoader):
    _df_columns: List[str] = [
        "unix",
        "date",
        "symbol",
        "open",
        "high",
        "low",
        "close",
        "Volume USDT",
        "tradecount",
    ]

    def __init__(self, config: DataLoaderConfig) -> None:
        super().__init__(config)

        # Load the data from disk and add missing timestamps
        self._df: pl.DataFrame = self._load_data()

        # Add missing timestamps
        self._df = self._add_missing_timestamps()

        # Set attributes needed for iterating through the data
        # ASSUMES A 1M GRAIN !!
        self._current_datetime = self._df["date"].dt.min() + timedelta(
            seconds=self._config.burn_in_seconds
        )
        self._idx_incr = self._config.interval_seconds // 60 * len(self.get_pairs())
        self._idx_diff = self._config.max_history_seconds // 60 * len(self.get_pairs())
        self._idx_from, self._idx_to = self._set_starting_idx()

    def _load_data(self):
        df = None
        for p in self._get_data_paths():

            logger.info(f"[HistoricalDataLoader] reading next file: {p}")
            df_next = pl.read_csv(
                p, columns=self._df_columns, skip_rows=1, parse_dates=True
            )

            if df is None:
                df = df_next
            else:
                df = pl.concat([df, df_next])

        df = df.sort(["date", "symbol"])
        return df

    def _set_starting_idx(self) -> Tuple[int, int]:

        idx_to = (self._df["date"] == self._current_datetime).arg_true().max() + 1
        idx_from = max(0, idx_to - self._idx_diff)

        return idx_from, idx_to

    def _increment_idx(self) -> None:

        self._idx_to += self._idx_incr
        self._idx_from += self._idx_incr

        self._current_datetime += timedelta(seconds=self._config.interval_seconds)
        logger.info(f"[data_factory] current date: {self._current_datetime}")

        if self._idx_to - self._idx_from != self._idx_diff:
            self._idx_from = max(0, self._idx_to - self._idx_diff)

    def _add_missing_timestamps(self):

        # !! ASSUMES A 1M GRAIN !!

        # Prepare skeleton with all timestamps
        df_skeleton = prepare_skeleton(
            from_datetime=self._df["date"].dt.min(),
            to_datetime=self._df["date"].dt.max(),
            interval="1M",
            pairs=self.get_pairs(),
        )

        # Join df to skeleton to keep missings explicitly
        # First truncate dates to 1M in case there are any sub-minute values in date
        pl_grain = to_polars_duration("1M")
        df = self._df.with_column(pl.col("date").dt.truncate(pl_grain))
        df = df_skeleton.join(df, on=["symbol", "date"], how="left")

        return df

    def _get_data_paths(self):

        # Prepare for matching
        parent_path = Path(self._config.hist__parent_path)
        exchange = self._config.hist__exchange.capitalize()
        pairs = [
            f"{base}{quote}"
            for base, quote in [split_pair(pair) for pair in self.get_pairs()]
        ]

        # Match pattern to historical data files
        candidates = [str(p) for p in parent_path.glob("*")]
        regex = re.compile(f".+{exchange}_({'|'.join(pairs)})_minute\.csv")
        paths = [Path(p).absolute() for p in filter(regex.match, candidates)]

        return paths

    def get_complete(self) -> DataFrameData:

        # Create data object
        data = DataFrameData(self._config)
        data.set_up(df=self._df)

        return data

    def data_factory(self) -> DataFrameData:

        if self._idx_to > self._df.shape[0]:
            return None

        # Filter out everything later than current_datetime and before max history period
        df = self._df[self._idx_from : self._idx_to]

        # Create data object
        data = DataFrameData(self._config)
        data.set_up(df=df)

        assert self._current_datetime == df["date"].dt.max(), "date not matching"

        # Increment current idx
        self._increment_idx()

        return data
