
import pandas as pd
import numpy as np
import copy
import pytest

from tradingtools import portfolio


start_positions = [
    {
        'symbol': 'BTCUSD',
        'timestamp_execution': pd.Timestamp("2017-01-01 12:00:00"),
        'volume': 10,
        'price_execution': 100,
        'timestamp_settlement': pd.Timestamp("2017-01-01 12:00:02"),
        'price_settlement': 101
    }, {
        'symbol': 'ETHUSD',
        'timestamp_execution': pd.Timestamp("2017-01-03 12:00:00"),
        'volume': 10,
        'price_execution': 120,
        'timestamp_settlement': pd.Timestamp("2017-01-03 12:00:02"),
        'price_settlement': 121
    }, {
        'symbol': 'ETHUSD',
        'timestamp_execution': pd.Timestamp("2017-01-10 12:00:00"),
        'volume': -10,
        'price_execution': 120,
        'timestamp_settlement': pd.Timestamp("2017-01-10 12:00:02"),
        'price_settlement': 101
    }, 
    {
        'symbol': 'BTCUSD',
        'timestamp_execution': pd.Timestamp("2017-01-21 12:00:00"),
        'volume': -5,
        'price_execution': 90,
        'timestamp_settlement': pd.Timestamp("2017-01-21 12:00:02"),
        'price_settlement': 87
    }
]

df_start = pd.DataFrame(start_positions)
pf = portfolio.Portfolio(df_start)


def test_add_order():

    pf2 = copy.deepcopy(pf)

    shape_before = pf2.get_orders().shape

    pf2.add_order(
        symbol="BTCUSD",
        timestamp_execution=pd.Timestamp("2017-01-30 12:00:00"),
        volume=10,
        price_execution=100,
        timestamp_settlement=pd.Timestamp("2017-01-30 12:00:02"),
        price_settlement=101
    )
    df = pf2.get_orders()
   
    assert df.shape == (shape_before[0] + 1, shape_before[1])
    assert df['symbol'].iloc[-1] == "BTCUSD"
    assert df['timestamp_execution'].iloc[-1] == pd.Timestamp("2017-01-30 12:00:00")
    assert df['volume'].iloc[-1] == 10
    assert df['price_execution'].iloc[-1] == 100

    with pytest.raises(Exception):
        pf2.add_order(
            symbol="BTCUSD",
            timestamp_execution=pd.Timestamp("2017-01-30 12:00:00"),
            volume=-100,
            price_execution=100,
            timestamp_settlement=pd.Timestamp("2017-01-30 12:00:02"),
            price_settlement=101
        )

def test_get_positions():

    pf2 = copy.deepcopy(pf)

    df_net = pf2.get_positions()
    print(df_net)
    
    assert df_net['volume'][0] == 5
    assert df_net['volume'][1] == 0

    # Add position to make position neutral
    pf2.add_order(
        symbol="BTCUSD",
        timestamp_execution=pd.Timestamp("2017-02-03 12:00:00"),
        volume=-5,
        price_execution=100,
        timestamp_settlement=pd.Timestamp("2017-02-03 12:00:02"),
        price_settlement=101
    )

    df_net = pf2.get_positions()
    
    assert df_net['volume'][0] == 0
    assert df_net['volume'][1] == 0


if __name__ == "__main__":
    test_get_positions()
    print(pf)