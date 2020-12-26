
import pandas as pd 

from tradingtools import utils

def test_timestamp_conversion():

    ts = pd.Timestamp.now()
    ts_str = utils.timestamp_to_string(ts)
    ts2 = utils.string_to_timestamp(ts_str)
    assert ts == ts2


if __name__ == "__main__":
    test_timestamp_conversion()

