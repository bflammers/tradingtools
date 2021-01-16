import pytest

import pandas as pd
import numpy as np

from decimal import Decimal

from tradingtools.broker import Order, Settlement
from tradingtools.portfolio import Portfolio


def test_initialize():

    price1 = Decimal(1000)
    amount1 = Decimal(100)
    amount2 = Decimal(200)

    pf = Portfolio(verbose=False)

    assert pf._starting_capital is None
    assert pf._unallocated_capital is None
    assert pf._reserved_capital == Decimal(0)
    assert pf.symbols == dict()

    tick = [
        {
            "trading_pair": "BTC/EUR",
            "close": Decimal(price1),
            "timestamp": pd.Timestamp.now(),
        }
    ]

    amounts = {"BTC": amount1, "EUR": amount2}

    pf.initialize(amounts=amounts, tick=tick)

    assert list(pf.symbols.keys()) == ["BTC"]
    assert pf._starting_capital == amount2 + price1 * amount1
    assert pf._unallocated_capital == amount2
    assert pf._reserved_capital == Decimal(0)

    pnl = pf.profit_and_loss()

    assert pnl["start_capital"] == amount2 + price1 * amount1
    assert pnl["unallocated"] == amount2
    assert pnl["total_value"] == amount2 + price1 * amount1
    assert pnl["profit_percentage"] == Decimal(0)
    assert list(pnl["symbols"].keys()) == ["BTC"]


def test_update():

    price1 = Decimal(1000)
    price2 = Decimal(2000)
    amount1 = Decimal(100)
    amount2 = Decimal(200)
    starting_capital = Decimal(10000)

    pf = Portfolio(verbose=False)

    amounts = {"BTC": Decimal(0), "EUR": Decimal(starting_capital)}

    tick = [
        {
            "trading_pair": "BTC/EUR",
            "close": Decimal(price1),
            "timestamp": pd.Timestamp.now(),
        }
    ]

    optimal_postions = {"BTC/EUR": amount1}

    with pytest.raises(Exception):
        # First need to initialize portfolio
        orders = pf.update(tick=tick, optimal_positions=optimal_postions)

    pf.initialize(amounts=amounts, tick=tick)

    orders = pf.update(tick=tick, optimal_positions=optimal_postions)
    order1 : Order = orders[0]

    assert len(orders) == 1
    assert order1.amount == amount1
    assert order1.side == "buy"
    assert order1.trading_pair == "BTC/EUR"
    assert order1.status == "pending"
    assert order1.cost_execution == amount1 * price1
    assert list(pf.symbols.keys()) == ["BTC"]
    assert pf._reserved_capital == amount1 * price1

    order_cost = amount1 * price1 * Decimal(1.001)
    settlement = Settlement(
        order_id=order1.order_id,
        status="completed",
        trading_pair=order1.trading_pair,
        value=amount1 * price1,
        cost=order_cost,
        price=price1,
        amount=amount1
    )

    pf.settle_order(settlement=settlement)
    pnl = pf.profit_and_loss()

    assert pnl['unallocated'] == starting_capital - order_cost


    tick = [
        {
            "trading_pair": "BTC/EUR",
            "close": Decimal(price2),
            "timestamp": pd.Timestamp.now(),
        }
    ]

    orders = pf.update(tick=tick, optimal_positions=optimal_postions)
    pnl = pf.profit_and_loss()
    print(pnl)

    assert len(orders) == 0
    assert pf
    

    optimal_postions = {"BTC/EUR": amount2}







if __name__ == "__main__":

    test_initialize()
    test_update()
