import logging

import pandas as pd
import numpy as np
import talib as ta

from .preprocessing import get_rolling_block

logger = logging.getLogger(__name__)

#### Helpers

def drop_na(X: pd.DataFrame, *args):
    
    na_rows = X.isna().any(axis=1)
    logger.info(f"Dropping {sum(na_rows)} rows because of na's in X")

    # Drop
    result = (X.loc[~na_rows.values], )
    for arg in args:
        result += (arg.loc[~na_rows.values], )
    
    return result


def drop_after_timegap(gaps: pd.Series, n: int, *args):
    
    # Determine indices (timestamps) that need to be dropped
    drop_mask = gaps.rolling(n).sum().shift() >= 1
    logger.info(f"Dropping {sum(drop_mask)} rows because of timegaps ({n} rows following gap dropped)")
    
    # Drop
    result = tuple()
    for arg in args:
        result += (arg.loc[~drop_mask], )
    
    return result


#### Pre processing


def rolling_ohlc(x: pd.Series, window: int = 60):
    rw = x.rolling(window=window)
    df = pd.concat([x.shift(periods=window - 1), rw.max(), rw.min(), x], axis=1)
    df.columns = ["open", "high", "low", "close"]
    df.iloc[: window - 1] = np.nan
    return df


def rolling_standardize(x: pd.Series, window: int = 100):

    if window < 4:
        logger.warning("rolling standardize with window < 4, returning unscaled")
        return x, pd.Series(0, index=x.index), pd.Series(1, index=x.index)

    # Rolling window, calc mean and std dev
    rw = x.rolling(window=window)
    mu = rw.mean()
    sd = rw.std()

    # Location and scale normalization
    sx = x.copy()
    sx -= mu
    sx /= sd

    return sx, mu, sd


def rolling_minmax_standardize(x: pd.Series, window: int = 100):

    if window < 4:
        logger.warning("rolling standardize with window < 4, returning unscaled")
        return x, pd.Series(0, index=x.index), pd.Series(1, index=x.index)

    # Rolling window, calc mean and std dev
    rw = x.rolling(window=window)
    mi = rw.min()
    ma = rw.max()

    # Calculcate scale, set to 1 if 0 for stability
    sc = ma - mi
    sc[sc == 0] = 1

    # Location and scale normalization
    sx = x.copy()
    sx -= mi
    sx /= sc

    return sx, mi, sc


##### Features


def first_diff(x: pd.Series, n: int):
    df = pd.DataFrame(index=x.index)
    col_names = ["first_diff_lag" + str(i) for i in range(n)[::-1]]
    df[col_names] = get_rolling_block(x.diff(), n, same_size=True)
    return df


def moving_average(x: pd.Series, n: int, nskip: int = 0):
    col_name = f"ma_n{n}{'_nskip' + str(nskip) if nskip else ''}"
    df = pd.DataFrame(index=x.index)
    df[col_name] = x.rolling(window=n).mean().shift(periods=nskip)
    return df


def bollinger(x: pd.Series, n: int):
    bb_upper, _, bb_lower = ta.BBANDS(x.values, timeperiod=n)

    df = pd.DataFrame(index=x.index)
    df["up_diff"] = bb_upper - x.values
    df["lo_diff"] = x.values - bb_lower
    df["squeeze"] = bb_upper - bb_lower

    return df


def sar(x: pd.Series, ohlc_window: int):
    ohlc = rolling_ohlc(x, ohlc_window)
    return ta.SAR(high=ohlc["high"], low=ohlc["low"])


def adx(x: pd.Series, ohlc_window: int, n: int):
    ohlc = rolling_ohlc(x, ohlc_window)
    return ta.ADX(high=ohlc["high"], low=ohlc["low"], close=ohlc["close"], timeperiod=n)


def boundary_levels(x: pd.Series, n: int, support_q: float, resist_q: float):

    rw = x.rolling(n)

    df = pd.DataFrame(index=x.index)
    df[f"sup_lvl_dist_n{n}_q{support_q}"] = x - rw.quantile(support_q)
    df[f"res_lvl_dist_n{n}_q{resist_q}"] = rw.quantile(1 - resist_q) - x

    return df
