from decimal import Decimal
from typing import AsyncIterator, List
from logging import getLogger

from .filler import AbstractFillStrategy

from ...assets import SymbolAsset
from ...utils import Order


logger = getLogger(__name__)


class MarketOrderFillStrategy(AbstractFillStrategy):
    async def _generate_orders(self, quantity: Decimal) -> AsyncIterator[Order]:

        # Set the target quantity for the base asset
        target_quantity_base = self._base_asset.get_quantity() + quantity

        # Calculate difference in quote value
        value_diff = self._base_asset.get_value_difference(
            quantity, self._quote_asset.get_name()
        )

        n_retries = 0

        while (
            self._check_tolerance(value_diff) and n_retries < self._config.max_retries
        ):

            order = Order(
                symbol=self._base_asset.get_name(),
                side="buy" if quantity_diff > Decimal("0.0") else "sell",
                quantity=quantity_diff,
                type="market",
            )

            yield order

            # Determine the current difference, update value_diff
            quantity_diff = target_quantity_base - self._base_asset.get_quantity()
            value_diff = self._base_asset.get_value_difference(
                quantity_diff, self._quote_asset.get_name()
            )

            n_retries += 1

        # TODO: something for cancelling all outstanding orders
