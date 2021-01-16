from numpy.core.records import fromfile
import pytest

import pandas as pd
import numpy as np

from decimal import Decimal

from tradingtools.utils import timestamp_to_string
from tradingtools.broker import Order, Settlement
from tradingtools.portfolio import Portfolio, Symbol


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


def test_sync():

    price1 = Decimal(1000)
    price2 = Decimal(2000)
    amount1 = Decimal(100)
    amount2 = Decimal(200)
    starting_capital = Decimal(10000)
    ts1 = timestamp_to_string(pd.Timestamp.now())
    ts2 = timestamp_to_string(pd.Timestamp.now())

    pf = Portfolio(verbose=False)

    amounts1 = {"BTC": Decimal(amount1), "EUR": Decimal(starting_capital)}

    tick1 = [
        {
            "trading_pair": "BTC/EUR",
            "close": price1,
            "timestamp": ts1,
        }
    ]
    pf.initialize(amounts=amounts1, tick=tick1)

    amounts2 = {"BTC": amount2, "EUR": Decimal(starting_capital)}
    pf.sync(amounts=amounts2)

    symbol = pf.symbols["BTC"]

    assert symbol._current_amount == amount2
    assert symbol._latest_price == price1
    assert symbol._latest_tick_timestamp == ts1
    assert symbol.get_current_value() == amount2 * price1

    tick2 = [
        {
            "trading_pair": "BTC/EUR",
            "close": price2,
            "timestamp": ts2,
        }
    ]

    pf.sync(tick=tick2)

    assert symbol._current_amount == amount2
    assert symbol._latest_price == price2
    assert symbol._latest_tick_timestamp == ts2
    assert symbol.get_current_value() == amount2 * price2

def test_update():

    price1 = Decimal(1000)
    price2 = Decimal(2000)
    amount1 = Decimal(0)
    amount2 = Decimal(200)
    starting_capital = Decimal(10000)

    pf = Portfolio(verbose=False)

    amounts = {"BTC": amount1, "EUR": Decimal(starting_capital)}

    tick = [
        {
            "trading_pair": "BTC/EUR",
            "close": Decimal(price1),
            "timestamp": pd.Timestamp.now(),
        }
    ]

    optimal_postions = {"BTC/EUR": amount2}

    with pytest.raises(Exception):
        # First need to initialize portfolio
        orders = pf.update(tick=tick, optimal_positions=optimal_postions)

    pf.initialize(amounts=amounts, tick=tick)

    orders = pf.update(tick=tick, optimal_positions=optimal_postions)
    order1: Order = orders[0]

    assert len(orders) == 1
    assert order1.amount == amount2
    assert order1.side == "buy"
    assert order1.trading_pair == "BTC/EUR"
    assert order1.status == "pending"
    assert order1.cost_execution == amount2 * price1
    assert list(pf.symbols.keys()) == ["BTC"]
    assert pf._reserved_capital == amount2 * price1

    symbol = pf.symbols['BTC']

    assert symbol._current_amount == amount1
    assert symbol._pending_delta_amount == amount2 
    assert len(symbol._open_orders) == 1

def test_settle_order():

    price1 = Decimal(1000)
    price2 = Decimal(2000)
    amount1 = Decimal(0)
    amount2 = Decimal(200)
    starting_capital = Decimal(10000)

    pf = Portfolio(verbose=False)

    amounts = {"BTC": amount1, "EUR": Decimal(starting_capital)}

    tick = [
        {
            "trading_pair": "BTC/EUR",
            "close": price1,
            "timestamp": pd.Timestamp.now(),
        }
    ]

    pf.initialize(amounts=amounts, tick=tick)

    optimal_postions = {"BTC/EUR": amount2}
    orders = pf.update(tick=tick, optimal_positions=optimal_postions)
    order1: Order = orders[0]

    order_cost = amount2 * price1 * Decimal(1.001)
    settlement = Settlement(
        order_id=order1.order_id,
        status="completed",
        trading_pair=order1.trading_pair,
        value=amount2 * price1,
        cost=order_cost,
        price=price1,
        amount=amount2,
    )

    pf.settle_order(settlement=settlement)

    assert pf._reserved_capital == Decimal(0)
    assert pf._unallocated_capital == starting_capital - order_cost

    symbol = pf.symbols['BTC']

    assert symbol.get_current_value() == price1 * amount2
    assert symbol._current_amount == amount2
    assert symbol._pending_delta_amount == Decimal(0)
    assert len(symbol._open_orders) == 0


if __name__ == "__main__":

    test_initialize()
    test_sync()
    test_update()
    test_settle_order()
