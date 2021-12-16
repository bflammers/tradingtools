import asyncio

from uuid import uuid4
from decimal import Decimal
from typing import List
from dataclasses import dataclass

from .broker import AbstractBroker
from .visitors import AbstractAssetVisitor


@dataclass
class AssetConfig:
    name: str
    tolerance_EUR: Decimal = Decimal(0.01)


class AbstractAsset:

    _config: AssetConfig = None
    _quantity: Decimal = None
    _broker: AbstractBroker = None

    def __init__(self, config: AssetConfig, broker: AbstractBroker = None) -> None:
        self._id = uuid4.hex()
        self._config = config
        self._broker = broker

    def update_price(self, prices) -> None:
        raise NotImplementedError

    async def update_quantity(self, quantities) -> None:
        raise NotImplementedError

    def accept(self, visitor: AbstractAssetVisitor) -> None:
        raise NotImplementedError

    def get_value(self) -> Decimal:
        raise NotImplementedError

    def get_quantity(self) -> Decimal:
        raise NotImplementedError

    def get_price(self) -> Decimal:
        raise NotImplementedError


class CompositeAsset(AbstractAsset):

    _children: List[AbstractAsset] = []

    def __init__(self, config: AssetConfig, broker: AbstractBroker = None) -> None:
        super().__init__(config, broker)
        self._quantity = Decimal(1)

    def update_price(self, prices) -> None:
        for child in self._children:
            child.update_price(prices)

    async def update_quantity(self, quantities) -> None:
        for child in self._children:
            await child.update_quantity(quantities)

    def accept(self, visitor: AbstractAssetVisitor) -> None:
        visitor.visit_composite_asset(self)
        for child in self._children:
            child.accept(visitor)

    def get_value(self) -> Decimal:
        value = Decimal(0)
        for child in self._children:
            value += child.get_value()

        return value

    def get_quantity(self) -> Decimal:
        return self._quantity

    def get_price(self) -> Decimal:
        return self.get_value()


class SymbolAsset(AbstractAsset):

    _price: Decimal
    _quantity_update_lock: asyncio.Lock = asyncio.Lock()

    def __init__(self, config: AssetConfig, broker: AbstractBroker = None) -> None:
        super().__init__(config, broker)
        self._quantity = Decimal(0)

    def update_price(self, prices) -> None:
        self._price = prices[self._name]

    async def update_quantity(self, quantities) -> None:
        new_quantity = quantities[self._name]

        async with self._quantity_update_lock:
            difference = new_quantity - self._quantity
            if difference > self._config.tolerance_EUR:
                await self._broker.order(difference, self)

    def accept(self, visitor: AbstractAssetVisitor) -> None:
        visitor.visit_symbol_asset(self)

    def get_value(self) -> Decimal:
        return self._quantity * self._price

    def get_quantity(self) -> Decimal:
        return self._quantity

    def get_price(self) -> Decimal:
        return self._price
