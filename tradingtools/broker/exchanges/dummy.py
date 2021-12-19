import asyncio

from logging import getLogger
from random import uniform
from datetime import datetime

from ...utils import Order
from .exchange import AbstractExchange, ExchangeConfig


logger = getLogger(__name__)


class DummyExchange(AbstractExchange):
    def _exchange_factory(self):
        return None

    async def place_order(self, order: Order) -> dict:

        logger.info(f"[DummyBroker] placing order {order.order_id}")

        await asyncio.sleep(uniform(0.2, 2.0))
        order_response = {
            "price": uniform(10.0, 1000.0),
            "amount": order.amount,
            "datetime": datetime.now().isoformat(),
            "id": "123abc",
            "fee": {"cost": uniform(1.0, 10.0), "currency": "EUR"},
        }

        logger.info(f"[DummyBroker] order {order.order_id} response: {order_response}")

        return order_response
