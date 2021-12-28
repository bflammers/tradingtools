import unittest
import asyncio
from unittest.mock import patch, MagicMock, AsyncMock
from time import time

from ccxt.async_support import Exchange

from tradingtools.broker import exchanges
from tradingtools.utils import Order


class TestExchange(unittest.TestCase):
    @patch("tradingtools.broker.exchanges.BinanceExchange._exchange_factory")
    def test_place_order(self, mock_exchange_factory: MagicMock):

        # Generate fake order
        fake_order = Order(
            symbol="AAA/BBB", side="buy", quantity=3.0, type="limit", price=999.0
        )

        # Side effect function that creates a fake order response
        def fake_order_response(symbol, type, side, amount, price, params):
            filled = amount * 0.9
            fee_cost = price * filled * 0.001
            return {
                "price": price,
                "cost": price * filled + fee_cost,
                "timestamp": 123456,
                "filled": filled,
                "id": "123abc",
                "fee": {"cost": fee_cost, "currency": "EUR"},
                "trades": ["xx"],
                "status": "xx",
            }

        # Mock the factory method so that it returns a mocked exchange
        mock_exchange = MagicMock(spec=Exchange)
        mock_exchange.create_order = AsyncMock(side_effect=fake_order_response)
        mock_exchange_factory.return_value = mock_exchange

        # Create the exchange
        config = exchanges.ExchangeConfig(
            type=exchanges.ExchangeTypes.binance,
            backtest=True,
            credentials={"api_key": "1234", "secret": "abcd"},
        )
        exchange = exchanges.BinanceExchange(config)

        # Place order and test if correct
        loop = asyncio.get_event_loop()
        order_response = asyncio.gather(exchange.place_order(fake_order))
        loop.run_until_complete(order_response)

        expected_order_response = fake_order_response(
            symbol=fake_order.symbol,
            type=fake_order.type,
            side=fake_order.side,
            amount=fake_order.quantity,
            price=fake_order.price,
            params={},
        )
        self.assertDictEqual(expected_order_response, order_response.result()[0])
