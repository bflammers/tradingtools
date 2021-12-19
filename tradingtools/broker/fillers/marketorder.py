from decimal import Decimal
from typing import AsyncIterator, List
from logging import getLogger

from .filler import AbstractFillStrategy

from ...assets import SymbolAsset
from ...utils import Order


logger = getLogger(__name__)


class MarketOrderFillStrategy(AbstractFillStrategy):
    async def _generate_orders(self, opt_quantity: Decimal) -> AsyncIterator[Order]:

        # Calculate difference, check if exceeds tolerance
        quantity_diff, price_diff = self._asset.get_difference(opt_quantity)

        while self._check_tolerance(price_diff):

            order = Order(
                symbol=self._asset.get_name(),
                side="buy" if quantity_diff > Decimal("0.0") else "sell",
                amount=quantity_diff,
                type="market",
            )

            yield order

            quantity_diff, price_diff = self._asset.get_difference(opt_quantity)
