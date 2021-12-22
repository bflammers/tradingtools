from decimal import Decimal
from typing import AsyncIterator, List
from logging import getLogger
from dataclasses import dataclass
from enum import Enum

from ..exchanges import AbstractExchange
from ...utils import Order, split_pair
from ...assets import SymbolAsset

logger = getLogger(__name__)


class FillerTypes(Enum):
    marketorder = "marketorder"


@dataclass
class FillStrategyConfig:
    type: FillerTypes
    difference_tol_EUR: Decimal = Decimal("1")


class AbstractFillStrategy:
    _base_asset: SymbolAsset
    _quote_asset: SymbolAsset
    _exchange: AbstractExchange
    _config: FillStrategyConfig

    def __init__(
        self,
        base_asset: SymbolAsset,
        quote_asset: SymbolAsset,
        exchange: AbstractExchange,
        config: FillStrategyConfig,
    ) -> None:
        self._base_asset = base_asset
        self._quote_asset = quote_asset
        self._exchange = exchange
        self._config = config

    async def fill(self, opt_quantity: Decimal) -> None:

        # Create and place orders, update assets
        async for order in self._generate_orders(opt_quantity):

            # Create and place orders
            order_response = self._exchange.place_order(order)
            settled_order = self._exchange.settle_order(order, order_response)

            # Update assets
            # TODO: make this transactional
            base_new = self._base_asset.get_quantity() + settled_order.amount_settlement
            quote_new = self._quote_asset.get_quantity() - settled_order.cost
            self._base_asset.set_quantity(base_new)
            self._quote_asset.set_quantity(quote_new)

    def _check_tolerance(self, price_diff: Decimal):

        if abs(price_diff) < self._config.difference_tol_EUR:
            logger.info(
                f"[FillStrategy] price diff of {price_diff} below tolerance for {self._base_asset.get_name()}"
            )
            return False

        return True

    async def _generate_orders(opt_quantity: Decimal) -> AsyncIterator[Order]:
        raise NotImplementedError
