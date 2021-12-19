from logging import getLogger

import ccxt.async_support as ccxt

from .exchange import AbstractExchange


logger = getLogger(__name__)


class BinanceExchange(AbstractExchange):

    _exchange_name: str = "binance"

    def _exchange_factory(self):
        exchange = ccxt.binance(
            {
                "apiKey": self._config.credentials["api_key"],
                "secret": self._config.credentials["secret_key"],
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
