from decimal import Decimal
import unittest

from tradingtools import assets


class TestTransaction(unittest.TestCase):
    def test_simple(self):

        A = assets.SymbolAsset(name="A", default_quote="C")
        B = assets.SymbolAsset(name="B", default_quote="C")

        A.set_quantity(quantity=Decimal("0.0"))
        B.set_quantity(quantity=Decimal("10.0"))

        self.assertEqual(A.get_quantity(), Decimal("0"))
        self.assertEqual(B.get_quantity(), Decimal("10"))

        quantity = Decimal("5.0")
        assets.AssetTransaction().add(A, quantity).subtract(B, quantity).commit()

        self.assertEqual(A.get_quantity(), Decimal("5"))
        self.assertEqual(B.get_quantity(), Decimal("5"))

    def test_rollback(self):

        A = assets.SymbolAsset(name="A", default_quote="C")
        B = assets.SymbolAsset(name="B", default_quote="C")

        A.set_quantity(quantity=Decimal("0.0"))
        B.set_quantity(quantity=Decimal("10.0"))

        self.assertEqual(A.get_quantity(), Decimal("0"))
        self.assertEqual(B.get_quantity(), Decimal("10"))

        quantity = Decimal("5.0")
        self.assertRaises(
            TypeError,
            assets.AssetTransaction().add(A, quantity).subtract(B, 5.0).commit,
        )

        self.assertEqual(A.get_quantity(), Decimal("0"))
        self.assertEqual(B.get_quantity(), Decimal("10"))

        quantity = Decimal("50.0")
        self.assertRaises(
            Exception,
            assets.AssetTransaction().add(A, quantity).subtract(B, quantity).commit,
        )

        self.assertEqual(A.get_quantity(), Decimal("0"))
        self.assertEqual(B.get_quantity(), Decimal("10"))


