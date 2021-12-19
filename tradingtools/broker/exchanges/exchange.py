import asyncio

from logging import getLogger
from decimal import Decimal
from dataclasses import dataclass

import ccxt.async_support as ccxt

from ...utils import Order


logger = getLogger(__name__)


@dataclass
class ExchangeConfig:
    type: str
    backtest: bool
    credentials: dict = {}
    backtest: bool = True


class AbstractExchange:

    _exchange: ccxt.Exchange
    _exchange_name: str = None

    def __init__(self, config: ExchangeConfig) -> None:
        self._config = config

        # Factories
        self._exchange = self._exchange_factory()

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
            amount=order.amount,
            price=order.price,
            params=params,
        )

        logger.info(f"[Broker] order {order.order_id} response: {order_response}")

        return order_response

    def settle_order(self, order: Order, order_response: dict):

        order.settle(
            price=Decimal(order_response["price"]),
            amount=Decimal(order_response["amount"]),
            timestamp=order_response["datetime"],
            exchange_order_id=order_response["id"],
            fee=Decimal(order_response["fee"]["cost"]),
            fee_currency=order_response["fee"]["currency"],
        )

        return order
