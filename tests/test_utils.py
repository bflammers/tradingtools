from decimal import Decimal
from time import time
import unittest
from tradingtools.broker.exchanges import exchange
import tradingtools.utils as utils


class TestFloatToDecimals(unittest.TestCase):
    def _test_helper(self, argument: float, expected: Decimal, precision: int):
        result = utils.float_to_decimal(argument, precision)
        self.assertEqual(result, expected)

    def test_integer(self):
        # Integer part
        self._test_helper(101, Decimal("101"), 3)
        self._test_helper(101.123, Decimal("101.123"), 3)
        self._test_helper(201.123, Decimal("201.123"), 3)

    def test_decimal(self):
        # Decimal part
        self._test_helper(101.0, Decimal("101"), 3)
        self._test_helper(101.12, Decimal("101.12"), 3)
        self._test_helper(101.123, Decimal("101.123"), 3)
        self._test_helper(101.1234, Decimal("101.123"), 3)
        self._test_helper(101.12345, Decimal("101.123"), 3)

    def test_precision(self):
        # Precision
        self._test_helper(101.1234567, Decimal("101"), 0)
        self._test_helper(101.1234567, Decimal("101.1"), 1)
        self._test_helper(101.1234567, Decimal("101.12"), 2)
        self._test_helper(101.1234567, Decimal("101.12346"), 5)


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
