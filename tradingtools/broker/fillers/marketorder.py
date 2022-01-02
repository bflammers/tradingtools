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

        if gap > Decimal("0.0"):  # Buy on market
            side = "buy"
            buy_side = self._base_asset
            sell_side = self._quote_asset
        else:  # Sell on market
            side = "sell"
            buy_side = self._quote_asset
            sell_side = self._base_asset

        n_retries = 0
        target = abs(gap)
        filled = Decimal("0")

        # Calculate open difference in quantity and value
        order_quantity, order_value = self._determine_diff(target, filled)

        while (
            self._continue_order(order_value) and n_retries < self._config.max_retries
        ):

            if order_value > sell_side.get_value():
                logger.warning(
                    f"[FillStrategy] not enough value in {sell_side.get_name()} to buy {order_value:0.3f} worth of {buy_side.get_name()}"
                )
                break

            order = Order(
                symbol=self._market,
                side=side,
                quantity=order_quantity,
                type="market",
                price=self._base_asset.get_price(),
            )

            yield order

            # Update open difference in quantity and value
            filled += order.filled_quantity
            order_quantity, order_value = self._determine_diff(target, filled)

            n_retries += 1
