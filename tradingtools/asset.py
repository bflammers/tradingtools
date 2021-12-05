
from uuid import uuid4
from decimal import Decimal
from typing import List


class Asset:

    _name: str = None
    _quantity: Decimal = None

    def __init__(self, name, broker) -> None:
        self._id = uuid4.hex()
        self._name = name
        self._broker = broker

    def update_price(self, prices):
        raise NotImplementedError

    def update_quantity(self, quantities):
        raise NotImplementedError

    def accept(self, visitor):
        raise NotImplementedError

    def get_value(self):
        raise NotImplementedError

    def get_quantity(self):
        raise NotImplementedError

    def get_price(self):
        raise NotImplementedError


class CompositeAsset(Asset):

    _children: List[Asset] = []

    def __init__(self, name, broker) -> None:
        super().__init__(name, broker)
        self._quantity = Decimal(1)

    def update_price(self, prices):
        for child in self._children:
            child.update_price(prices)

    def update_quantity(self, quantities):
        for child in self._children:
            child.update_quantity(quantities)

    def accept(self, visitor):
        for child in self._children:
            child.accept(visitor)

    def get_value(self):
        value = Decimal(0)
        for child in self._children:
            value += child.get_value()

        return value

    def get_quantity(self):
        return self._quantity

    def get_price(self):
        return self.get_value()

class SymbolAsset(Asset):

    _price : Decimal
    _quantity_lock: bool = False

    def __init__(self, name, broker) -> None:
        super().__init__(name, broker)
        self._quantity = Decimal(0)

    def update_price(self, prices):
        self._price = prices[self._name]

    def update_quantity(self, quantities):
        new_quantity = quantities[self._name]
        if self._quantity != new_quantity:
            if not self._quantity_locked():
                difference = new_quantity - self._quantity
                self._broker.order(difference, self._update_quantity_callback)

    def accept(self, visitor):
        visitor.visit_symbol(self)

    def get_value(self):
        return self._quantity * self._price

    def get_quantity(self):
        return self._quantity

    def get_price(self):
        return self._price

    def _update_quantity_callback(self, quantity):
        self._quantity = quantity
        self._quantity_lock = False

    def _quantity_locked(self):
        if self._quantity_lock:
            return True
        
        self._quantity_lock = True
        return False