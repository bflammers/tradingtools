from __future__ import annotations  # For correct typing when returning self
from logging import getLogger
from uuid import uuid4
from decimal import Decimal
from typing import List, Dict, Tuple
from time import time

from .visitors import AbstractAssetVisitor
from .utils import split_pair

logger = getLogger(__name__)


class AbstractCompositeAsset:
    def __init__(
        self, name: str, default_quote: str, time_diff_tol_sec: float = 120.0
    ) -> None:
        self._id = uuid4().hex
        self._name = name
        self._default_quote = default_quote
        self._time_diff_tol_sec = time_diff_tol_sec
        self._quantity: Decimal = None
        self._price_update_time: time = None

    def get_name(self) -> str:
        return self._name

    def get_asset_names(self) -> List[str]:
        raise NotImplementedError

    def set_quantity(self, quantity: Decimal) -> None:
        raise NotImplementedError

    def get_quantity(self) -> Decimal:
        return self._quantity

    def set_price(self, price: Decimal, quote: str = None) -> None:
        raise NotImplementedError

    def update_prices(self, prices: Dict[str, Decimal]) -> None:
        raise NotImplementedError

    def get_price(self, quote: str = None) -> Decimal:
        raise NotImplementedError

    def get_value(self, quote: str = None) -> Decimal:
        quote = quote or self._default_quote
        return self._quantity * self.get_price(quote)

    def get_value_difference(self, quantity: Decimal, quote: str) -> Tuple[Decimal]:
        quote = quote or self._default_quote
        quantity_diff = quantity - self.get_quantity()
        return quantity_diff * self.get_price(quote)

    def get_market(self, quote: str = None) -> str:
        raise NotImplementedError

    def accept(self, visitor: AbstractAssetVisitor) -> None:
        raise NotImplementedError


class PortfolioAsset(AbstractCompositeAsset):
    def __init__(
        self, name: str, default_quote: str, time_diff_tol_sec: float = 120
    ) -> None:
        super().__init__(name, default_quote, time_diff_tol_sec=time_diff_tol_sec)
        self._children: List[AbstractCompositeAsset] = []
        self._quantity: Decimal = Decimal("1")

    def get_asset_names(self) -> List[str]:
        return [child.get_name() for child in self._children]

    def update_prices(self, prices: Dict[str, Decimal]) -> None:
        for child in self._children:
            child.update_prices(prices)

    def get_price(self, quote: str = None) -> Decimal:
        quote = quote or self._default_quote
        return self.get_value(quote)

    def get_value(self, quote: str = None) -> Decimal:
        quote = quote or self._default_quote
        value = Decimal("0")
        for child in self._children:
            value += child.get_value(quote)
        return value

    def accept(self, visitor: AbstractAssetVisitor) -> None:
        visitor.visit_composite_asset(self)
        for child in self._children:
            child.accept(visitor)

    def add_asset(self, asset: AbstractCompositeAsset) -> None:
        logger.info(f"[CompositeAsset.add_asset] adding asset {asset.get_name()}")
        self._children.append(asset)

    def get_asset(self, name: str) -> AbstractCompositeAsset:

        for child in self._children:
            if child.get_name() == name:
                return child

        logger.warning(f"[CompositeAsset.get_asset] no child with name {name}")
        return None

    def __repr__(self) -> str:
        summary = f"Total value: {self.get_value():.3f} {self._default_quote}\n"
        for child in self._children:
            summary += child.__repr__()
        return summary


class SymbolAsset(AbstractCompositeAsset):
    def __init__(
        self, name: str, default_quote: str, time_diff_tol_sec: float = 120
    ) -> None:
        super().__init__(name, default_quote, time_diff_tol_sec=time_diff_tol_sec)
        self._price: Dict[str, dict] = {}
        self._quantity: Decimal = Decimal("0")

    def set_quantity(self, quantity: Decimal) -> None:
        if quantity < Decimal("0"):
            message = (
                f"[Asset] cannot set negative quantity {quantity} for {self._name}"
            )
            logger.error(message)
            raise Exception(message)

        self._quantity = Decimal(quantity)

    def update_prices(self, prices: Dict[str, Decimal]) -> None:
        for name, price in prices.items():
            base, quote = split_pair(name)
            if base == self._name:
                self.set_price(price, quote)

    def get_price(self, quote: str = None) -> Decimal:

        quote = quote or self._default_quote

        try:
            price_dict = self._price[quote]
        except KeyError:
            logger.warning(f"[Asset] price not set for {self._name}/{quote}")
            return None

        time_diff = time() - price_dict["update_time"]
        if self._time_diff_tol_sec and time_diff > self._time_diff_tol_sec:
            logger.warning(
                f"[Asset] price for {self._name}/{quote} not updated for {time_diff} seconds"
            )

        return price_dict["price"]

    def set_price(self, price: Decimal, quote: str = None) -> None:
        if price < Decimal("0"):
            message = f"[Asset] cannot set negative price {price} for {self._name}"
            logger.error(message)
            raise Exception(message)

        quote = quote or self._default_quote
        self._price[quote] = {"price": price, "update_time": time()}

    def get_value(self, quote: str = None) -> Decimal:
        quote = quote or self._default_quote
        return self._quantity * self.get_price(quote)

    def get_market(self, quote: str = None) -> str:
        quote = quote or self._default_quote

        if not quote:
            logger.warning(
                f"[SymbolAsset.get_default_market] no default quote for {self._name}"
            )

        return f"{self._name}/{self._default_quote}"

    def accept(self, visitor: AbstractAssetVisitor) -> None:
        visitor.visit_symbol_asset(self)

    def __repr__(self) -> str:
        summary = f"-- {self.get_name():<7}"
        summary += f" >> {'quantity: ':<10}{round(self.get_quantity(), 5):>12} "
        summary += (
            f" >> {'price: ':<10}{round(self.get_price(), 3):>8} {self._default_quote}"
        )
        summary += f" >> {'value: ':<10}{round(self.get_value(), 3):>8} {self._default_quote}\n"
        return summary


class AssetTransaction:
    def __init__(self) -> None:
        self.sub_transactions: List[Tuple[AbstractCompositeAsset, Decimal]] = []
        self.rollback_quantities: List[Decimal] = []

    def add(self, asset: AbstractCompositeAsset, quantity: Decimal) -> AssetTransaction:
        self._create_sub_transaction(asset, quantity)
        return self

    def subtract(
        self, asset: AbstractCompositeAsset, quantity: Decimal
    ) -> AssetTransaction:
        self._create_sub_transaction(asset, -quantity)
        return self

    def _create_sub_transaction(
        self, asset: AbstractCompositeAsset, quantity: Decimal
    ) -> AssetTransaction:
        self.sub_transactions.append((asset, quantity))
        self.rollback_quantities.append(None)

    def _apply_modification(self, index: int) -> None:

        # Extract asset, and quantities from sub transaction to apply
        asset, addition_quantity = self.sub_transactions[index]
        current_quantity = asset.get_quantity()

        # Save current quantity for rollback
        self.rollback_quantities[index] = current_quantity

        # Set new quantity
        new_quantity = current_quantity + addition_quantity
        asset.set_quantity(new_quantity)

        if asset.get_quantity() != new_quantity:
            raise Exception(
                f"[AssetTransaction] quantity not correctly set to new value for {asset.get_name()}"
            )

    def _rollback_modification(self, index: int) -> None:

        # Extract asset, and quantities from sub transaction to apply
        asset, _ = self.sub_transactions[index]
        rollback_quantity = self.rollback_quantities[index]
        asset.set_quantity(rollback_quantity)

        if asset.get_quantity() != rollback_quantity:
            raise Exception(
                f"[AssetTransaction] rollback failed for {asset.get_name()}"
            )

    def _rollback(self, index: int):

        failed = False
        failed_assets = []
        for i in range(index, -1, -1):

            try:
                self._rollback_modification(i)
            except:
                failed = True
                asset = self.sub_transactions[i][0]
                failed_assets.append(asset.get_name())

        if failed:
            message = f"[AssetTransaction] rollback failed for assets: {failed_assets}"
            logger.error(message)
            raise Exception(message)

    def commit(self):

        for i in range(len(self.sub_transactions)):

            try:
                self._apply_modification(i)
            except:

                logger.error(
                    "[AssetTransaction] transaction failed, attempting rollback"
                )
                self._rollback(i)
                logger.info("[AssetTransaction] rollback successful")

                raise
