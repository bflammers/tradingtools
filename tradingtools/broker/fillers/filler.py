from decimal import Decimal
from typing import AsyncIterator, Dict, Tuple
from logging import getLogger
from dataclasses import dataclass, field
from enum import Enum

from ..exchanges import AbstractExchange
from ...utils import Order, split_pair
from ...assets import SymbolAsset, AssetTransaction

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
    def __init__(
        self,
        config: FillStrategyConfig,
        base_asset: SymbolAsset,
        quote_asset: SymbolAsset,
        exchange: AbstractExchange,
    ) -> None:
        self._config: FillStrategyConfig = config
        self._base_asset: SymbolAsset = base_asset
        self._quote_asset: SymbolAsset = quote_asset
        self._exchange: AbstractExchange = exchange

        # Safe base and quote asset names, determine market
        self._base_name: str = self._base_asset.get_name()
        self._quote_name: str = self._quote_asset.get_name()
        self._market: str = f"{self._base_name}/{self._quote_name}"

    async def fill(self, gap: Decimal) -> None:

        # Create and place orders, update assets
        async for order in self._generate_orders(gap):

            # Create and place order with exchange
            order_response = await self._exchange.place_order(order)

            # Update the local order object
            updated_order = self._exchange.update_order(order, order_response)

            # Update assets
            # TODO: what happens when an order is filled in pieces?
            # TODO: should quantity be reserved for each asset? --> not with cancelling outstanding
            (
                AssetTransaction()
                .add(self._base_asset, updated_order.filled_quantity)
                .subtract(self._quote_asset, updated_order.filled_quantity)
                .commit()
            )

    def _continue_order(self, value_diff: Decimal):

        # Check if there is enough value in quote asset
        if value_diff > self._quote_asset.get_value():
            logger.warning(
                f"[FillStrategy] not enough value in {self._quote_asset.get_name()} to buy {value_diff:0.3f} worth of {self._base_asset.get_name()}"
            )
            return False

        # Safe get the tolerance
        quote_name = self._quote_asset.get_name()
        try:
            tolerance = self._config.value_diff_tol_quote[quote_name]
        except KeyError:
            logger.warning(
                f"[FillStrategy] no value_diff_tol_quote for {quote_name} - skipping check"
            )
            return True

        # Compare price difference to tolerance
        if abs(value_diff) < tolerance:
            logger.info(
                f"[FillStrategy] value diff of {value_diff:0.3f} below tolerance for {self._market}"
            )
            return False

        return True

    def _determine_diff(self, target: Decimal, filled: Decimal) -> Tuple[Decimal]:
        quantity_diff = target - filled
        value_diff = quantity_diff * self._base_asset.get_price(self._quote_name)
        return quantity_diff, value_diff

    async def _generate_orders(quantity: Decimal) -> AsyncIterator[Order]:
        raise NotImplementedError
