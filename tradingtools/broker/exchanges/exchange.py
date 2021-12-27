import asyncio

from pprint import pformat
from logging import getLogger
from decimal import Decimal
from dataclasses import dataclass
from enum import Enum
from datetime import datetime

import ccxt.async_support as ccxt

from ...utils import Order


logger = getLogger(__name__)


class ExchangeTypes(Enum):
    dummy = "dummy"
    binance = "binance"


@dataclass
class ExchangeConfig:
    type: ExchangeTypes
    backtest: bool
    credentials: dict = None
    backtest: bool = True


class AbstractExchange:

    _exchange_name: str = None

    def __init__(self, config: ExchangeConfig) -> None:
        self._config = config

        # Factories
        self._exchange: ccxt.Exchange = self._exchange_factory()

        # Checks
        self._live_trading_confirmation()

    def _live_trading_confirmation(self):

        if not self._config.backtest:
            confirmation = input(
                "[Broker] backtest = False, are you sure you want to perform actual trades? (y)"
            )

            if confirmation != "y":
                raise Exception("Live trading aborted")

    def _exchange_factory(self):
        raise NotImplementedError(
            "[AbstractExchange] not implemented for abstract class"
        )

    async def place_order(self, order: Order) -> dict:

        params = {
            "test": self._config.backtest,  # test if it's valid, but don't actually place it
        }

        logger.info(f"[Broker] placing order {order.order_id}")

        # Create market order through ccxt with specified exchange
        order_response = await self._exchange.create_order(
            symbol=order.symbol,
            type=order.type,
            side=order.side,
            amount=order.quantity,
            price=order.price,
            params=params,
        )

        logger.debug(f"[Broker] order {order.order_id} response: \n{pformat(order_response)}")

        return order_response

    def update_order(self, order: Order, order_response: dict):

        order.update(
            price=Decimal(order_response["price"]),
            cost=Decimal(order_response["cost"]),
            timestamp=datetime.fromtimestamp(order_response["timestamp"] / 1000),
            filled_quantity=Decimal(order_response["filled"]),
            exchange_order_id=order_response["id"],
            fee=Decimal(order_response["fee"]["cost"]),
            fee_currency=order_response["fee"]["currency"],
            trades=order_response["trades"],
            status=order_response["status"],
        )

        return order
