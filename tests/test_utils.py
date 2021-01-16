
import pandas as pd 

from tradingtools.utils import timestamp_to_string, string_to_timestamp

def test_timestamp_conversion():

    ts = pd.Timestamp.now()
    ts_str = timestamp_to_string(ts)
    ts2 = string_to_timestamp(ts_str)
    assert ts == ts2


if __name__ == "__main__":
    test_timestamp_conversion()

