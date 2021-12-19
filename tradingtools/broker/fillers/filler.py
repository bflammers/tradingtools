from decimal import Decimal
from typing import AsyncIterator, List
from logging import getLogger
from dataclasses import dataclass

from ..exchanges import AbstractExchange
from ...utils import Order
from ...assets import SymbolAsset

logger = getLogger(__name__)


@dataclass
class FillStrategyConfig:
    type: str
    difference_tol_EUR: Decimal = Decimal("1")


class AbstractFillStrategy:
    _asset: SymbolAsset
    _exchange: AbstractExchange
    _config: FillStrategyConfig

    def __init__(
        self, asset: SymbolAsset, exchange: AbstractExchange, config: FillStrategyConfig
    ) -> None:
        self._asset = asset
        self._exchange = exchange
        self._config = config

    async def fill(self, opt_quantity: Decimal) -> None:

        # Create and place orders, update assets
        async for order in self._generate_orders(opt_quantity):

            # Create and place orders
            order_response = self._exchange.place_order(order)
            settled_order = self._exchange.settle_order(order, order_response)

            # Update assets
            new_quantity = self._asset.get_value() + settled_order.amount_settlement
            self._asset.set_quantity(new_quantity)

    def _check_tolerance(self, price_diff: Decimal):

        if price_diff < self._config.difference_tol_EUR:
            logger.info(
                f"[FillStrategy] price diff of {price_diff} below tolerance for {self._asset.get_name()}"
            )
            return False

        return True

    async def _generate_orders(opt_quantity: Decimal) -> AsyncIterator[Order]:
        raise NotImplementedError
