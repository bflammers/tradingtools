from logging import getLogger
from uuid import uuid4
from decimal import Decimal
from typing import List, Dict, Tuple
from time import time

from .visitors import AbstractAssetVisitor
from .utils import split_pair

logger = getLogger(__name__)

# TODO: implement transaction with rollback


class AbstractCompositeAsset:

    _name: str
    _price_update_time: time
    _time_diff_tol_sec: float
    _default_quote: str
    _quantity: Decimal = None

    def __init__(self, name: str, default_quote: str, time_diff_tol_sec: float = 120.0) -> None:
        self._id = uuid4().hex
        self._name = name
        self._default_quote = default_quote
        self._time_diff_tol_sec = time_diff_tol_sec

    def get_name(self) -> str:
        return self._name

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

    def accept(self, visitor: AbstractAssetVisitor) -> None:
        raise NotImplementedError


class PortfolioAsset(AbstractCompositeAsset):

    _children: List[AbstractCompositeAsset] = []
    _quantity: Decimal = Decimal("1")

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


class SymbolAsset(AbstractCompositeAsset):

    _quantity: Decimal = Decimal("0")
    _price: Dict[str, dict] = {}

    def set_quantity(self, quantity: Decimal) -> None:
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

        time_diff = time() - price_dict["update_time"]
        if self._time_diff_tol_sec and time_diff > self._time_diff_tol_sec:
            logger.warning(
                f"[Asset] price for {self._name}/{quote} not updated for {time_diff} seconds"
            )

        return price_dict["price"]

    def set_price(self, price: Decimal, quote: str) -> None:
        self._price[quote] = {"price": price, "update_time": time()}

    def get_value(self, quote: str = None) -> Decimal:
        quote = quote or self._default_quote
        return self._quantity * self.get_price(quote)

    def accept(self, visitor: AbstractAssetVisitor) -> None:
        visitor.visit_symbol_asset(self)
