from decimal import Decimal
from typing import AsyncIterator, Dict
from logging import getLogger
from dataclasses import dataclass, field
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
    value_diff_tol_quote: Dict[str, Decimal] = field(
        default_factory=lambda: {
            "EUR": Decimal("2.0"),
            "USD": Decimal("2.0"),
            "USDT": Decimal("2.0"),
        }
    )
    max_retries: int = 4


class AbstractFillStrategy:
    _base_asset: SymbolAsset
    _quote_asset: SymbolAsset
    _exchange: AbstractExchange
    _config: FillStrategyConfig
    _market: str

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
        self._market = f"{self._base_asset.get_name()}/{self._quote_asset.get_name()}"

    async def fill(self, quantity: Decimal) -> None:

        # Create and place orders, update assets
        async for order in self._generate_orders(quantity):

            # Create and place order with exchange
            order_response = self._exchange.place_order(order)

            # Update the local order object
            updated_order = self._exchange.update_order(order, order_response)

            # Update assets
            # TODO: make this transactional
            base_new = self._base_asset.get_quantity() + updated_order.filled_quantity
            quote_new = self._quote_asset.get_quantity() - updated_order.cost_settlement
            self._base_asset.set_quantity(base_new)
            self._quote_asset.set_quantity(quote_new)

    def _check_tolerance(self, price_diff: Decimal):

        # Get tolerance
        quote_name = self._quote_asset.get_name()
        try:
            tolerance = self._config.value_diff_tol_quote[quote_name]
        except KeyError:
            logger.warning(
                f"[FillStrategy] no value_diff_tol_quote for {quote_name} - skipping check"
            )
            return True

        # Compare price difference to tolerance
        if abs(price_diff) < tolerance:
            logger.info(
                f"[FillStrategy] price diff of {price_diff} below tolerance for {self._market}"
            )
            return False

        return True

    async def _generate_orders(quantity: Decimal) -> AsyncIterator[Order]:
        raise NotImplementedError
