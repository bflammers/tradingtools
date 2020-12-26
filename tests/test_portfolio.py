from numpy.core.fromnumeric import shape
import pandas as pd
import numpy as np

from tradingtools.portfolio import Portfolio


def test_update():

    # Init empty portfolio

    pf = Portfolio(start_capital=100)

    # Add single position

    optimal_positions = [{"symbol": "BTCUSD", "volume": 1}]

    prices = {
        "BTCUSD": 10,
        "ETHUSD": 20,
    }

    pf.update(prices, optimal_positions)
    positions = pf.get_optimal_positions()
    orders = pf.get_orders()

    assert positions.shape == (1, 3)
    assert orders.shape[0] == 1
    np.testing.assert_equal(
        orders[["symbol", "price_execution", "order_type", "volume"]].values,
        np.array([["BTCUSD", 10.0, "buy", 1.0]], dtype=np.object),
    )

    # Add second position

    optimal_positions = [
        {"symbol": "BTCUSD", "volume": 1},
        {"symbol": "ETHUSD", "volume": 2},
    ]

    orders = pf.update(prices, optimal_positions)
    positions = pf.get_optimal_positions()
    orders = pf.get_orders()

    assert positions.shape == (2, 4)
    assert orders.shape[0] == 2
    np.testing.assert_equal(
        orders[["symbol", "price_execution", "order_type", "volume"]].values,
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

    pf.update(prices, optimal_positions)
    positions = pf.get_optimal_positions()
    orders = pf.get_orders()

    # print("\npositions:\n", positions)
    # print("\norders:\n", orders)

    assert positions.shape == (3, 4)
    assert orders.shape[0] == 4
    np.testing.assert_equal(
        orders[["symbol", "price_execution", "order_type", "volume"]].values,
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
        positions[["BTCUSD", "ETHUSD"]].values[-1], np.array([0.0, 0.0])
    )


def test_full_update_cycle():

    # Init empty portfolio

    pf = Portfolio(start_capital=100)

    # Add single position

    optimal_positions = [
        {"symbol": "BTCUSD", "volume": 1},
        {"symbol": "ETHUSD", "volume": 4},
    ]

    prices = {
        "BTCUSD": 10,
        "ETHUSD": 20,
    }

    orders = pf.update(prices, optimal_positions)

    for order in orders:
        pf.add_settlement(
            order["order_id"], prices[order["symbol"]], pd.Timestamp.now()
        )

    pnl = pf.profit_and_loss()

    assert pnl["start_capital"] == 100
    assert pnl['unallocated'] == 10
    assert pnl['total_value'] == 100
    assert pnl['profit_percentage'] == 0

    prices = {
        "BTCUSD": 20,
        "ETHUSD": 40,
    }

    orders = pf.update(prices)
    pnl = pf.profit_and_loss()

    assert pnl["start_capital"] == 100
    assert pnl['unallocated'] == 10
    assert pnl['total_value'] == 190
    assert pnl['profit_percentage'] == 90




if __name__ == "__main__":

    test_update()
    test_full_update_cycle()
