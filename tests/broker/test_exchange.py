import unittest
import asyncio
from unittest.mock import patch, MagicMock, AsyncMock
from time import time

from ccxt.async_support import Exchange

from tradingtools.broker import exchanges
from tradingtools.utils import Order, RunType

dummy_config = exchanges.ExchangeConfig(
    type=exchanges.ExchangeTypes.dummy, run_type=RunType.dry_run
)


binance_config = exchanges.ExchangeConfig(
    type=exchanges.ExchangeTypes.binance,
    run_type=True,
    credentials={"api_key": "1234", "secret": "abcd"},
)


def fake_order():
    return Order(symbol="AAA/BBB", side="buy", quantity=100.0, type="limit", price=5.0)


def fake_order_response(symbol, type, side, amount, price, params):
    filled = amount * 0.9
    fee_cost = price * filled * 0.001
    return {
        "price": price,
        "cost": price * filled + fee_cost,
        "timestamp": 1640714067000,
        "filled": filled,
        "id": "123abc",
        "fee": {"cost": fee_cost, "currency": "EUR"},
        "trades": ["xx"],
        "status": "xx",
    }


class TestExchange(unittest.TestCase):
    @patch("tradingtools.broker.exchanges.BinanceExchange._exchange_factory")
    def test_place_order(self, mock_exchange_factory: MagicMock):

        # Generate fake order
        order = fake_order()

        # Mock the factory method so that it returns a mocked exchange
        mock_exchange = MagicMock(spec=Exchange)
        mock_exchange.create_order = AsyncMock(side_effect=fake_order_response)
        mock_exchange_factory.return_value = mock_exchange

        # Create the exchange
        exchange = exchanges.BinanceExchange(binance_config)

        # Place order and test if correct
        loop = asyncio.get_event_loop()
        order_response = asyncio.gather(exchange.place_order(order))
        loop.run_until_complete(order_response)

        expected_order_response = fake_order_response(
            symbol=order.symbol,
            type=order.type,
            side=order.side,
            amount=order.quantity,
            price=order.price,
            params={},
        )
        self.assertDictEqual(expected_order_response, order_response.result()[0])

    def test_update_order(self):

        # Fake order and order response
        order = fake_order()
        order_response = fake_order_response(
            symbol=order.symbol,
            type=order.type,
            side=order.side,
            amount=order.quantity,
            price=order.price,
            params={},
        )

        # Modify some values
        order_response.update({
            "price": 1.0,
            "cost": 2.0
        })

        # Create dummy exchange, update order
        exchange = exchanges.DummyExchange(dummy_config)
        updated_order = exchange.update_order(
            order=order, order_response=order_response
        )

        # Test 
        self.assertEqual(order_response["price"], order.price_settlement)
        self.assertEqual(order_response["price"], updated_order.price_settlement)
        self.assertEqual(order_response["cost"], order.cost_settlement)
        self.assertEqual(order_response["cost"], updated_order.cost_settlement)


if __name__ == "__main__":

    t = TestExchange()
    t.test_update_order()
