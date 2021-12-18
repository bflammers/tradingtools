
from decimal import Decimal
from typing import Dict


class AbstractFillStrategy:

    def __init__(self) -> None:
        pass

    def create_orders(self, symbol: str, difference: Decimal, price: Decimal):
        pass
