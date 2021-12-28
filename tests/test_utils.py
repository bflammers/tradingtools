from decimal import Decimal
from time import time
import unittest
from tradingtools.broker.exchanges import exchange
import tradingtools.utils as utils


class TestSplitPair(unittest.TestCase):
    def test_slash(self):

        base, quote = utils.split_pair("AAA/BBB")
        self.assertEqual(base, "AAA")
        self.assertEqual(quote, "BBB")

    def test_dash(self):
        base, quote = utils.split_pair("AAA-BBB")
        self.assertEqual(base, "AAA")
        self.assertEqual(quote, "BBB")

    def test_error(self):
        self.assertRaises(Exception, lambda: utils.split_pair("AAA.BBB"))


class TestOrder(unittest.TestCase):
    def test_order_creation(self):

        order = utils.Order(
            symbol="AAA/BBB", side="buy", quantity=Decimal("1"), type="market"
        )

        self.assertTrue(abs(order.timestamp_created - time()) < 1.0)
        self.assertIsNotNone(order.order_id)

        self.assertRaises(
            Exception,
            lambda: utils.Order(
                symbol="AAA/BBB", side="buy", quantity=Decimal("1"), type="limit"
            ),
        )

    def test_update(self):

        order = utils.Order(
            symbol="AAA/BBB",
            side="buy",
            quantity=1,
            type="market",
            status="open",
            price=1,
            timestamp_created=time(),
            order_id="abc",
            price_settlement=1,
            cost_settlement=1,
            timestamp_settlement=time(),
            filled_quantity=1,
            exchange_order_id="123",
            fee=1,
            fee_currency="EUR",
            trades=[],
        )

        order.update(price=2)
        self.assertEqual(order.price_settlement, 2)
        self.assertEqual(order.status, "settled")

        order.update(
            price=2,
            cost=3,
            timestamp=1000,
            filled_quantity=4,
            exchange_order_id="qwerty",
            fee=0.1,
            fee_currency="USDT",
            trades=[1, 2],
            status="open",
        )

        self.assertEqual(order.price_settlement, 2)
        self.assertEqual(order.cost_settlement, 3)
        self.assertEqual(order.timestamp_settlement, 1000)

