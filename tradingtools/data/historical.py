import re
import polars as pl

from logging import currentframe, getLogger
from decimal import Decimal
from typing import Dict, List
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
        self._current_datetime = None

    def set_up(self, df: pl.DataFrame, current_datetime: datetime) -> None:
        self._current_datetime = current_datetime
        self._df = self._add_missing_timestamps(df)

    def _add_missing_timestamps(self, df: pl.DataFrame):

        # assumes a 1M grain in data

        # Prepare skeleton with all timestamps
        df_skeleton = prepare_skeleton(
            from_datetime=df["date"].dt.min(),
            to_datetime=self._current_datetime,
            interval="1M",
            pairs=self.get_pairs(),
        )

        # Join df to skeleton to keep missings explicitly
        # First truncate dates to 1M in case there are any sub-minute values in date
        pl_grain = to_polars_duration("1M")
        df = df.with_column(pl.col("date").dt.truncate(pl_grain))
        df = df_skeleton.join(df, on=["symbol", "date"], how="left")

        return df

    def get_latest(self) -> Dict[str, Decimal]:

        if not self._config.interval_seconds:
            df_latest = self._df.groupby("symbol").agg(pl.col("close").last())
        else:

            # Filter out everything outside of tolerance period
            df = filter_period(self._df, length_seconds=self._config.update_tol_seconds)

            # Select last known closing prices
            df_latest = df.groupby("symbol").agg(pl.col("close").drop_nulls().last())

        # Collect pairs and prices, convert prices to decimal
        pairs = df_latest["symbol"].to_list()
        prices = df_latest["close_last"].to_list()
        prices = [float_to_decimal(price) for price in prices]

        current_date = df["date"].dt.max().isoformat()
        logger.info(f"[get_latest] current date: {current_date}")

        return dict(zip(pairs, prices))

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
        self._df: pl.DataFrame = self._load_data()
        self._min_datetime = self._df["date"].dt.min()
        self._max_datetime = self._df["date"].dt.max()
        self._current_datetime = self._get_start_datetime()

    def _get_start_datetime(self) -> datetime:
        start_datetime = self._min_datetime + timedelta(
            seconds=self._config.burn_in_seconds
        )

        if start_datetime > self._max_datetime:
            raise ValueError(
                f"[_get_start_datetime] start_datetime {start_datetime} is later than max_datetime {self._max_datetime}"
            )
        return start_datetime

    def _prepare_data_skeleton(self, df: pl):
        pass

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

    def data_factory(self) -> DataFrameData:

        if self._current_datetime > self._max_datetime:
            return None

        # Filter out everything later than current_datetime and before max history period
        df = filter_period(
            self._df,
            to_datetime=self._current_datetime,
            length_seconds=self._config.max_history_seconds,
        )

        # Create data object
        data = DataFrameData(self._config)
        data.set_up(df=df, current_datetime=self._current_datetime)

        # Increment current datetime
        self._current_datetime += timedelta(seconds=self._config.interval_seconds)

        return data
