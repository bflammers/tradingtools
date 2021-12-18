import asyncio

from logging import getLogger
from typing import Dict, List
from uuid import uuid4
from decimal import Decimal
from dataclasses import dataclass
from datetime import datetime
from random import randint

import ccxt.async_support as ccxt

from .assets import AbstractAsset


logger = getLogger(__name__)


@dataclass
class Order:

    # Execution
    symbol: str
    side: str
    amount: Decimal
    type: str
    status: str = "open"
    price: Decimal = None
    timestamp_created: datetime = datetime.now()
    order_id: str = uuid4().hex
    # Settlement
    price_settlement: Decimal = None
    timestamp_settlement: datetime = None
    exchange_order_id: str = None
    fee: Decimal = None
    fee_currency: str = None

    def __post_init__(self):

        if type == "market" and self.price is not None:
            logger.warning("[Broker] order type is market and price is not None")

    def settle(
        self,
        price: Decimal = None,
        timestamp: datetime = None,
        exchange_order_id: str = None,
        fee: Decimal = None,
        fee_currency: str = None,
    ) -> None:

        self.price_settlement = price
        self.timestamp_settlement = timestamp
        self.exchange_order_id = exchange_order_id
        self.fee = fee
        self.fee_currency = fee_currency
        self.status = "settled"


class AbstractBroker:

    _exchange: ccxt.Exchange 
    _exchange_name: str = None
    _assets: Dict[str: AbstractAsset]
    _orders: Dict[str: List[Order]]

    def __init__(self, credentials, backtest=True) -> None:
        self._credentials = credentials
        self._backtest = backtest

        self._exchange = self._exchange_factory()
        self._live_trading_confirmation()

    def _live_trading_confirmation(self):

        if not self._backtest:
            confirmation = input(
                "[Broker] backtest = False, are you sure you want to perform actual trades? (y)"
            )

            if confirmation != "y":
                raise Exception("Live trading aborted")

    @staticmethod
    def _exchange_factory(credentials: dict):
        raise NotImplementedError

    # def update_quantity(self, quantities) -> None:
    #     new_quantity = quantities[self._name]

    #     async with self._quantity_update_lock:
    #         difference = new_quantity - self._quantity
    #         if difference > self._config.tolerance_EUR:
    #             await self._broker.fill_position(difference, self)

    async def place_order(
        self, 
        order: Order
    ) -> dict:

        params = {
            "test": self._backtest,  # test if it's valid, but don't actually place it
        }

        # Create market order through ccxt with specified exchange
        order_response = await self._exchange.create_order(
            symbol=order.symbol,
            type=order.type,
            side=order.side,
            amount=order.amount,
            price=order.price,
            params=params,
        )

        logger.info(f"[Broker] order response: {order_response}")

        return order_response

    @staticmethod
    def _settle_order(order: Order, order_response: dict):

        order.settle(
            order_id=order.order_id,
            status="pending",
            trading_pair=order.trading_pair,
            exchange_order_id=order_response["id"],
            timestamp=order_response["datetime"],
            price=Decimal(order_response["price"]),
            value=Decimal(order_response["price"]) * Decimal(order_response["amount"]),
            amount=Decimal(order_response["amount"]),
            cost=Decimal(order_response["cost"]),
            average_cost=Decimal(order_response["average"]),
            fee=Decimal(order_response["fee"]["cost"]),
            fee_currency=order_response["fee"]["currency"],
        )


class BinanceBroker(AbstractBroker):

    _exchange_name: str = "binance"

    @staticmethod
    def _exchange_factory(credentials):
        exchange = ccxt.binance(
            {
                "apiKey": credentials["api_key"],
                "secret": credentials["secret_key"],
                "timeout": 30000,
                "enableRateLimit": True,
                "options": {
                    "adjustForTimeDifference": True,  # resolves the recvWindow timestamp error
                    "recvWindow": 59999,  # resolves the recvWindow timestamp error
                },
            }
        )

        exchange_status = exchange.fetch_status()["status"]
        logger.info(f"[BinanceBroker] logged into Binance -- Status: {exchange_status}")

        return exchange


class DummyBroker(AbstractBroker):
    def __init__(self, credentials=None, backtest=True) -> None:
        pass

    async def order(
        self,
        symbol: str,
        side: str,
        amount: Decimal,
        order_type: str,
        price: Decimal = None,
    ) -> dict:
        await asyncio.sleep(randint(1, 3))
