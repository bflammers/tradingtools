
from logging import getLogger
from uuid import uuid4
from decimal import Decimal
from typing import List, Dict
from time import time

from .visitors import AbstractAssetVisitor
from .utils import split_pair

logger = getLogger(__name__)


class Prices:

    time_tol_sec: float
    data: Dict[str, Dict[str, dict]] = {}

    def __init__(self, time_tol_sec: float = 120.0) -> None:
        self.time_tol_sec = time_tol_sec

    def clear(self) -> None:
        self.data = {}

    def _safe_get(self, base: str, quote: str) -> dict:

        try:
            base_dict = self.data[base]
        except KeyError:
            self.data[base] = {}
            base_dict = self.data[base]

        try:
            quote_dict = base_dict[quote]
        except KeyError:
            self.data[base][quote] = {}
            quote_dict = self.data[base][quote]

        return quote_dict

    def _safe_set(self, base: str, quote: str, price: Decimal) -> dict:

        _ = self._safe_get(base, quote)
        self.data[base][quote] = {
            "price": Decimal(price),
            "update_time": time(),
            "pair": f"{base}/{quote}",
        }

    def update(self, pair, price):
        base, quote = split_pair(pair)
        self._safe_set(base, quote, price)

    def get(self, pair: str) -> Decimal:
        base, quote = split_pair(pair)
        price = self._safe_get(base, quote)

        if price:
            time_diff = time() - price["update_time"]
            if time_diff > self.time_tol_sec:
                logger.warning(
                    f"[Prices] asset_name {pair} not updated for {time_diff} seconds"
                )
        else:
            logger.warning(f"[Prices] asset_name {pair} not yet present")
            return None

        return price["price"]


class AbstractAsset:

    _name: str
    _prices: Prices
    _quantity: Decimal = None

    def __init__(self, name: str, prices: Prices) -> None:
        self._id = uuid4.hex()
        self._name = name
        self._prices = prices

    def accept(self, visitor: AbstractAssetVisitor) -> None:
        raise NotImplementedError

    def get_value(self) -> Decimal:
        raise NotImplementedError

    def set_quantity(self, quantity: Decimal) -> None:
        raise NotImplementedError

    def get_quantity(self) -> Decimal:
        raise NotImplementedError

    def get_price(self) -> Decimal:
        raise NotImplementedError


class CompositeAsset(AbstractAsset):

    _children: List[AbstractAsset] = []
    _quantity: Decimal = Decimal('1')

    def accept(self, visitor: AbstractAssetVisitor) -> None:
        visitor.visit_composite_asset(self)
        for child in self._children:
            child.accept(visitor)

    def get_value(self) -> Decimal:
        value = Decimal('0')
        for child in self._children:
            value += child.get_value()

        return value

    def set_quantity(self, quantity: Decimal) -> None:
        logger.warning("[CompositeAsset] set_quantity() called on object of class CompositeAsset")

    def get_quantity(self) -> Decimal:
        return self._quantity

    def get_price(self) -> Decimal:
        return self.get_value()

    def add_asset(self, asset: AbstractAsset) -> None:
        self._children.append(asset)


class SymbolAsset(AbstractAsset):

    _price: Decimal
    _quantity: Decimal = Decimal('0')

    def accept(self, visitor: AbstractAssetVisitor) -> None:
        visitor.visit_symbol_asset(self)

    def get_value(self) -> Decimal:
        price = self.get_price()
        return self._quantity * price

    def set_quantity(self, quantity: Decimal) -> None:
        self._quantity = Decimal(quantity)

    def get_quantity(self) -> Decimal:
        return self._quantity

    def get_price(self) -> Decimal:
        return self._prices.get(self._name)
