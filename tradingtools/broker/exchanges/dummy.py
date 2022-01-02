import asyncio

from pprint import pformat
from logging import getLogger
from random import uniform
from time import time

from ...utils import Order, RunType
from .exchange import AbstractExchange, ExchangeConfig


logger = getLogger(__name__)


class DummyExchange(AbstractExchange):
    def __init__(self, config: ExchangeConfig) -> None:
        super().__init__(config)

        if self._config.run_type is RunType.live:
            message = "[DummyExchange] run_type == live not possible with DummyExchange"
            logger.error(message)
            raise ValueError(message)

    def _exchange_factory(self):
        return None

    async def place_order(self, order: Order) -> dict:

        logger.debug(
            f"[DummyExchange] placing order {order.order_id}: \n{pformat(order)}"
        )

        if self._config.run_type is RunType.dry_run:
            await asyncio.sleep(uniform(0.2, 2.0))

        # Generate fake order response
        price = float(order.price) or uniform(10.0, 1000.0)
        fee_cost = price * 0.001

        if order.type == "limit":
            remaining = order.quantity - order.filled_quantity
            filled = float(order.filled_quantity) + uniform(0.5, 1.0) * float(remaining)
        else:
            filled = float(order.quantity)

        order_response = {
            "price": price,
            "cost": price * filled + fee_cost,
            "timestamp": time() * 1000,
            "filled": filled,
            "id": "123abc",
            "fee": {"cost": fee_cost, "currency": "EUR"},
            "trades": ["xx"],
            "status": "xx",
        }

        logger.debug(
            f"[DummyExchange] order {order.order_id} response: \n{pformat(order_response)}"
        )

        return order_response
