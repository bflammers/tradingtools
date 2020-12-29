from numpy.core.fromnumeric import shape
import pandas as pd
import numpy as np

from tradingtools.portfolio import Portfolio


def test_update():

    # Init empty portfolio

    pf = Portfolio(start_capital=100)

    # Add single position

    optimal_positions = [{"symbol": "BTCUSD", "volume": 1}]

    tick = [
        {
            "symbol": "BTCUSD",
            "close": 10.0,
            "timestamp": "2019-01-02 23:25:00",
        },
        {
            "symbol": "ETHUSD",
            "close": 20.0,
            "timestamp": "2019-01-02 23:25:00",
        },
    ]

    orders = pf.update(tick=tick, optimal_positions=optimal_positions)
    df_positions = pf.get_optimal_positions()
    df_orders = pf.get_orders()

    assert df_positions.shape == (1, 3)
    assert df_orders.shape[0] == 1
    np.testing.assert_equal(
        df_orders[
            ["symbol", "status", "price_execution", "order_type", "volume"]
        ].values,
        np.array([["BTCUSD", "pending", 10.0, "buy", 1.0]], dtype=np.object),
    )

    pf.add_settlement(
        order_id=orders[0]["order_id"],
        price=11.0,
        fee=11 / 100,
        timestamp_settlement="2019-01-02 23:25:00",
    )
    df_orders = pf.get_orders()

    assert df_orders.shape[0] == 1
    np.testing.assert_equal(
        df_orders[
            [
                "symbol",
                "status",
                "price_execution",
                "order_type",
                "volume",
                "price_settlement",
                "slippage",
            ]
        ].values,
        np.array([["BTCUSD", "filled", 10.0, "buy", 1.0, 11.0, 1.0]], dtype=np.object),
    )

    # Add second position

    optimal_positions = [
        {"symbol": "BTCUSD", "volume": 1},
        {"symbol": "ETHUSD", "volume": 2},
    ]

    pf.update(tick, optimal_positions)
    df_positions = pf.get_optimal_positions()
    df_orders = pf.get_orders()

    assert df_positions.shape == (2, 4)
    assert df_orders.shape[0] == 2
    np.testing.assert_equal(
        df_orders[["symbol", "price_execution", "order_type", "volume"]].values,
        np.array(
            [["BTCUSD", 10.0, "buy", 1.0], ["ETHUSD", 20.0, "buy", 2.0]],
            dtype=np.object,
        ),
    )

    # Set both positions to zero

    optimal_positions = [
        {"symbol": "BTCUSD", "volume": 0},
        {"symbol": "ETHUSD", "volume": 0},
    ]

    pf.update(tick, optimal_positions)
    df_positions = pf.get_optimal_positions()
    df_orders = pf.get_orders()

    # print("\npositions:\n", positions)
    # print("\norders:\n", orders)

    assert df_positions.shape == (3, 4)
    assert df_orders.shape[0] == 4
    np.testing.assert_equal(
        df_orders[["symbol", "price_execution", "order_type", "volume"]].values,
        np.array(
            [
                ["BTCUSD", 10.0, "buy", 1.0],
                ["ETHUSD", 20.0, "buy", 2.0],
                ["BTCUSD", 10.0, "sell", 1.0],
                ["ETHUSD", 20.0, "sell", 2.0],
            ],
            dtype=np.object,
        ),
    )

    np.testing.assert_equal(
        df_positions[["BTCUSD", "ETHUSD"]].values[-1], np.array([0.0, 0.0])
    )


def test_full_update_cycle():

    # Init empty portfolio

    pf = Portfolio(start_capital=100)

    # Add single position

    optimal_positions = [
        {"symbol": "BTCUSD", "volume": 1},
        {"symbol": "ETHUSD", "volume": 4},
    ]

    tick = [
        {
            "symbol": "BTCUSD",
            "close": 10.0,
            "timestamp": "2019-01-02 23:25:00",
        },
        {
            "symbol": "ETHUSD",
            "close": 20.0,
            "timestamp": "2019-01-02 23:25:00",
        },
    ]

    orders = pf.update(tick, optimal_positions)

    for i, order in enumerate(orders):
        settlement_price = tick[i]["close"]
        pf.add_settlement(
            order_id=order["order_id"],
            price=settlement_price,
            fee=settlement_price / 100,
            timestamp_settlement=pd.Timestamp.now(),
        )

    pnl = pf.profit_and_loss()

    assert pnl["start_capital"] == 100
    assert pnl["unallocated"] == 10
    assert pnl["total_value"] == 100
    assert pnl["profit_percentage"] == 0

    tick = [
        {
            "symbol": "BTCUSD",
            "close": 20.0,
            "timestamp": "2019-01-02 23:25:00",
        },
        {
            "symbol": "ETHUSD",
            "close": 40.0,
            "timestamp": "2019-01-02 23:25:00",
        },
    ]

    orders = pf.update(tick)
    pnl = pf.profit_and_loss()

    assert pnl["start_capital"] == 100
    assert pnl["unallocated"] == 10
    assert pnl["total_value"] == 190
    assert pnl["profit_percentage"] == 90


if __name__ == "__main__":

    test_update()
    test_full_update_cycle()
