import pytest
from decimal import Decimal

from tradingtools.symbol import Symbol


def test_sync_state():

    ts = "2021-01-13 20:22:15.139162"
    price1 = Decimal(100)
    price2 = Decimal(200)
    amount1 = Decimal(10)
    amount2 = Decimal(20)

    s = Symbol("BTC")

    s.sync_state(tick_timestamp="2021-01-13 20:22:15.139162", price=price1)

    assert s._latest_price == price1
    assert s._latest_tick_timestamp == ts

    s.sync_state(current_amount=amount1)
    current_value = s.get_current_value()

    assert s._total_value_at_buy == amount1 * price1
    assert current_value == amount1 * price1
    assert s._current_amount == amount1

    s.sync_state(current_amount=amount2)
    current_value = s.get_current_value()

    assert s._total_value_at_buy == amount2 * price1
    assert current_value == amount2 * price1
    assert s._current_amount == amount2

    s.sync_state(tick_timestamp=ts, price=price2)
    current_value = s.get_current_value()

    assert s._latest_price == price2
    assert current_value == amount2 * price2


def test_update_optimal_positions():

    ts = "2021-01-13 20:22:15.139162"
    price1 = Decimal(100)
    amount1 = Decimal(10)
    amount2 = Decimal(20)

    s = Symbol("BTC")

    with pytest.raises(Exception):
        order = s.update_optimal_position(amount1)

    s.sync_state(tick_timestamp=ts, price=price1)

    order = s.update_optimal_position(amount1)

    # Check order
    assert {"order_id", "trading_pair", "side", "amount"}.issubset(set(order.keys()))

    assert order["side"] == "buy"
    assert order["amount"] == amount1
    assert order["price_execution"] == price1

    # Check position
    assert s.optimal_amount == amount1

    order = s.update_optimal_position(amount2)

    # Check order
    assert isinstance(order["order_id"], str)
    assert order["side"] == "buy"
    assert order["amount"] == abs(amount1 - amount2)
    assert order["price_execution"] == price1

    # Check position
    assert s.optimal_amount == amount2

    order = s.update_optimal_position(amount2)

    assert order is None


def test_add_settlement():

    ts = "2021-01-13 20:22:15.139162"
    price1 = Decimal(100)
    amount1 = Decimal(10)
    order_value1 = Decimal(20)

    s = Symbol("BTC")

    s.sync_state(tick_timestamp=ts, price=price1)

    order = s.update_optimal_position(amount1)

    assert s._current_amount == Decimal(0)
    assert s.get_current_value() == Decimal(0)
    assert s._pending_delta_amount == amount1

    s.add_settlement(order_id=order["order_id"], order_value=order_value1)

    assert s._current_amount == amount1
    assert s.get_current_value() == amount1 * price1
    assert s._total_value_at_buy == order_value1
    assert s._pending_delta_amount == 0


def test_profit_and_loss():

    ts = "2021-01-13 20:22:15.139162"
    price1 = Decimal(100)
    price2 = Decimal(200)
    amount1 = Decimal(10)
    amount2 = Decimal(20)
    order_value1 = Decimal(30)
    order_value2 = Decimal(40)

    s = Symbol("BTC")

    s.sync_state(tick_timestamp=ts, price=price1)
    order = s.update_optimal_position(amount1)
    s.add_settlement(order_id=order["order_id"], order_value=order_value1)

    pnl = s.profit_and_loss()

    assert pnl["amount"] == amount1
    assert pnl["value"] == amount1 * price1
    assert pnl["profit"] == amount1 * price1 - order_value1

    s.sync_state(tick_timestamp=ts, price=price2)
    order = s.update_optimal_position(amount2)
    s.add_settlement(order_id=order["order_id"], order_value=order_value2)

    pnl = s.profit_and_loss()

    assert pnl["amount"] == amount2
    assert pnl["value"] == amount2 * price2
    assert pnl["profit"] == amount2 * price2 - order_value1 - order_value2


if __name__ == "__main__":

    test_sync_state()
    test_update_optimal_positions()
    test_add_settlement()
    test_profit_and_loss()
