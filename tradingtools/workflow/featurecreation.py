import pandas as pd
import numpy as np


def rolling_ohlc(x: pd.Series, window=60):
    rw = x.rolling(window=window)
    df = pd.concat([x.shift(periods=window - 1), rw.max(), rw.min(), x], axis=1)
    df.columns = ["open", "high", "low", "close"]
    df.iloc[: window - 1] = np.nan
    return df
