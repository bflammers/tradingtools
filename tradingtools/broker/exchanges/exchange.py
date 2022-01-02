import asyncio

from pprint import pformat
from logging import getLogger
from decimal import Decimal
from dataclasses import dataclass
from enum import Enum
from datetime import datetime

from ccxt.async_support import Exchange

from ...utils import Order, RunType, float_to_decimal


logger = getLogger(__name__)


class ExchangeTypes(Enum):
    dummy = "dummy"
    binance = "binance"


@dataclass
class ExchangeConfig:
    type: ExchangeTypes
    credentials: dict = None
    run_type: RunType = None  # Set by BrokerConfig

    def __post_init__(self):

        if self.run_type is RunType.live:
            self._live_trading_confirmation()

    def _live_trading_confirmation(self):

        confirmation = input(
            "[Broker] RunType is LIVE, are you sure you want to perform actual trades? (y)"
        )

        if confirmation != "y":
            raise Exception("Live trading aborted")


class AbstractExchange:

    exchange: Exchange
    _exchange_name: str = None

    def __init__(self, config: ExchangeConfig) -> None:
        self._config = config

        # Factories
        self.exchange: Exchange = self._exchange_factory()

    def _exchange_factory(self):
        raise NotImplementedError(
            "[AbstractExchange] not implemented for abstract class"
        )

    async def place_order(self, order: Order) -> dict:

        params = {"test": False if self._config.run_type is RunType.live else True}

        logger.debug(f"[Broker] placing order {order.order_id}")

        # Create market order through ccxt with specified exchange
        order_response = await self.exchange.create_order(
            symbol=order.symbol,
            type=order.type,
            side=order.side,
            amount=order.quantity,
            price=order.price,
            params=params,
        )

        logger.debug(
            f"[Broker] order {order.order_id} response: \n{pformat(order_response)}"
        )

        return order_response

    def update_order(self, order: Order, order_response: dict):

        order.update(
            price=float_to_decimal(order_response["price"]),
            cost=float_to_decimal(order_response["cost"]),
            timestamp=datetime.fromtimestamp(order_response["timestamp"] / 1000),
            filled_quantity=float_to_decimal(order_response["filled"]),
            exchange_order_id=order_response["id"],
            fee=float_to_decimal(order_response["fee"]["cost"]),
            fee_currency=order_response["fee"]["currency"],
            trades=order_response["trades"],
            status=order_response["status"],
        )

        return order

    # TODO: something for cancelling all outstanding orders
