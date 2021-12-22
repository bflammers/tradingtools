from logging import getLogger
from uuid import uuid4
from decimal import Decimal
from typing import List, Dict, Tuple
from time import time

from .utils import split_pair

logger = getLogger(__name__)


class AbstractMarket:

    _pair: str
    _price_update_time: time
    _time_diff_tol_sec: float
    _price: Decimal = None

    def __init__(self, pair: str, time_diff_tol_sec: float = 120.0) -> None:
        self._id = uuid4().hex
        self._pair = pair
        self._time_diff_tol_sec = time_diff_tol_sec

    def get_name(self) -> str:
        return self._pair

    def set_quantity(self, quantity: Decimal) -> None:
        raise NotImplementedError

    def get_quantity(self) -> Decimal:
        return self._quantity

    def set_price(self, price: Decimal) -> None:
        raise NotImplementedError

    def update_prices(self, prices: Dict[str, Decimal]) -> None:
        raise NotImplementedError

    def get_price(self) -> Decimal:
        raise NotImplementedError

    def get_value(self) -> Decimal:
        price = self.get_price()
        return self._quantity * price

    def get_difference(self, quantity: Decimal) -> Tuple[Decimal]:
        quantity_diff = quantity - self.get_quantity()
        value_diff = quantity_diff * self.get_price()
        return quantity_diff, value_diff

    def accept(self, visitor: AbstractAssetVisitor) -> None:
        raise NotImplementedError


class CompositeAsset(AbstractAsset):

    _children: List[AbstractAsset] = []
    _quantity: Decimal = Decimal("1")

    def update_prices(self, prices: Dict[str, Decimal]) -> None:
        for child in self._children:
            child.update_prices(prices)

    def get_price(self) -> Decimal:
        return self.get_value()

    def get_value(self) -> Decimal:
        value = Decimal("0")
        for child in self._children:
            value += child.get_value()
        return value

    def accept(self, visitor: AbstractAssetVisitor) -> None:
        visitor.visit_composite_asset(self)
        for child in self._children:
            child.accept(visitor)

    def add_asset(self, asset: AbstractAsset) -> None:
        logger.info(f"[CompositeAsset.add_asset] adding asset {asset.get_name()}")
        self._children.append(asset)

    def get_asset(self, name: str) -> AbstractAsset:

        for child in self._children:
            if child.get_name() == name:
                return child

        logger.warning(f"[CompositeAsset.get_asset] no child with name {name}")
        return None


class SymbolAsset(AbstractAsset):

    _quantity: Decimal = Decimal("0")

    def set_quantity(self, quantity: Decimal) -> None:
        self._quantity = Decimal(quantity)

    def update_prices(self, prices: Dict[str, Decimal]) -> None:
        new_price = prices[self.get_name()]
        self.set_price(new_price)

    def get_price(self) -> Decimal:

        if not self._price:
            logger.warning(f"[Asset.get_price] price not set for {self._name}")

        time_diff = time() - self._price_update_time
        if time_diff > self._time_diff_tol_sec:
            logger.warning(
                f"[Prices] asset_name {self._name} not updated for {time_diff} seconds"
            )

        return self._price

    def set_price(self, price: Decimal) -> None:
        self._price = price
        self._price_update_time = time()

    def get_value(self) -> Decimal:
        price = self.get_price()
        return self._quantity * price

    def accept(self, visitor: AbstractAssetVisitor) -> None:
        visitor.visit_symbol_asset(self)
