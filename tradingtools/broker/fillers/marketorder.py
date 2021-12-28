import asyncio
from decimal import Decimal
from typing import AsyncIterator, List
from logging import getLogger

from .filler import AbstractFillStrategy

from ...assets import SymbolAsset
from ...utils import Order


logger = getLogger(__name__)


class MarketOrderFillStrategy(AbstractFillStrategy):
    async def _generate_orders(self, gap: Decimal) -> AsyncIterator[Order]:

        n_retries = 0
        filled = Decimal("0")

        # Calculate open difference in quantity and value
        quantity_diff, value_diff = self._determine_diff(gap, filled)

        while self._continue_order(value_diff) and n_retries < self._config.max_retries:

            if value_diff > self._quote_asset.get_value():
                logger.warning(f"")

            side = "buy" if quantity_diff > Decimal("0.0") else "sell"

            # Wait 1 second for sell orders to come through
            if side == "buy":
                await asyncio.sleep(1)

            order = Order(
                symbol=self._base_asset.get_name(),
                side=side,
                quantity=quantity_diff,
                type="market",
                price=self._base_asset.get_price(),
            )

            yield order

            # Update open difference in quantity and value
            filled += order.filled_quantity
            quantity_diff, value_diff = self._determine_diff(gap, filled)

            n_retries += 1

