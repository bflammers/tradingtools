from datetime import datetime, timedelta
from typing import List

import polars as pl

from ..utils import split_interval


def to_polars_duration(interval: str):
    quantity, unit = split_interval(interval)
    if unit not in ["D", "H", "M", "S"]:
        raise ValueError(f"[to_polars_duration] unit {unit} not supported")
    return f"{quantity}{unit.lower()}"


def prepare_skeleton(
    from_datetime: datetime,
    to_datetime: datetime,
    interval: str,
    pairs: List[str],
    time_unit: str = "ms",
) -> pl.DataFrame:

    # Prepare timestamps
    pl_duration = to_polars_duration(interval)
    timestamps = pl.date_range(
        low=from_datetime, high=to_datetime, interval=pl_duration, time_unit=time_unit
    )

    # Repeat for all pairs
    df_skel = None
    for pair in pairs:
        df_next = pl.DataFrame(
            {"date": timestamps, "symbol": pl.repeat(pair, len(timestamps))}
        )

        if df_skel is None:
            df_skel = df_next
        else:
            df_skel = pl.concat([df_skel, df_next])

    # Sort 
    df_skel = df_skel.sort(["date", "symbol"])

    return df_skel


def filter_period(
    df: pl.DataFrame, to_datetime: datetime = None, length_seconds: int = None
) -> pl.DataFrame:

    # Using unix timestamp for filtering is not noticebly faster

    if to_datetime is None and length_seconds is None:
        raise ValueError(
            "[filter_period] at least one of to_datetime and length_seconds should be not None"
        )

    if to_datetime is None:
        to_datetime = df["date"].dt.max()
    else:
        df = df.filter(pl.col("date") <= to_datetime)

    if length_seconds is not None:
        from_datetime = to_datetime - timedelta(seconds=length_seconds)
        df = df.filter(pl.col("date") >= from_datetime)

    return df

