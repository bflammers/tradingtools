import asyncio

from pprint import pformat
from logging import getLogger
from random import uniform
from time import time

from ...utils import Order
from .exchange import AbstractExchange, ExchangeConfig


logger = getLogger(__name__)


class DummyExchange(AbstractExchange):
    def _exchange_factory(self):
        return None

    async def place_order(self, order: Order) -> dict:

        logger.info(f"[DummyBroker] placing order {order.order_id}")

        await asyncio.sleep(uniform(0.2, 2.0))

        # Generate fake order response
        price = float(order.price) or uniform(10.0, 1000.0)
        fee_cost = price * 0.0001
        filled = float(order.quantity) * uniform(0.5, 1.0)
        order_response = {
            "price": price,
            "cost": price * filled + fee_cost,
            "timestamp": time() * 1000,
            "filled": filled,
            "id": "123abc",
            "fee": {"cost": fee_cost, "currency": "EUR"},
            "trades": ["xx"], 
            "status": "xx"
        }

        logger.debug(f"[DummyBroker] order {order.order_id} response: \n{pformat(order_response)}")

        return order_response
