import asyncio
from asyncio.futures import Future

from logging import getLogger
from typing import Dict
from decimal import Decimal
from dataclasses import dataclass

from tradingtools.broker.exchanges import exchange

from ..assets import AbstractCompositeAsset, PortfolioAsset
from ..utils import Order, split_pair
from .fillers import AbstractFillStrategy, FillStrategyConfig, filler_factory
from .exchanges import AbstractExchange, ExchangeConfig, exchange_factory


logger = getLogger(__name__)


@dataclass
class BrokerConfig:
    backtest: bool
    filler__config: FillStrategyConfig
    exchange__config: ExchangeConfig


# TODO: still need a sync method to synchronize assets with exchange state
class Broker:

    _exchange: AbstractExchange

    def __init__(self, config: BrokerConfig) -> None:
        self._config: BrokerConfig = config
        self._fillers: Dict[str, AbstractFillStrategy] = {}

        # Factories
        self._exchange = exchange_factory(self._config.exchange__config)

    def _get_or_create_filler(self, market: str, portfolio: PortfolioAsset):

        try:
            return self._fillers[market]
        except KeyError:

            # Get base and quote assets
            base, quote = split_pair(market)
            base_asset = portfolio.get_asset(base)
            quote_asset = portfolio.get_asset(quote)

            # Create filler for this market
            self._fillers[market] = filler_factory(
                config=self._config.filler__config,
                base_asset=base_asset,
                quote_asset=quote_asset,
                exchange=self._exchange,
            )

            return self._fillers[market]

    async def fill(
        self, market_gaps: Dict[str, Decimal], portfolio: PortfolioAsset
    ) -> Future:

        fill_coroutines = []

        for market, gap in market_gaps.items():

            filler = self._get_or_create_filler(market, portfolio)

            # Append to list of coroutines
            fill_coroutines.append(filler.fill(gap))

        futures = await asyncio.gather(*fill_coroutines)

        return futures
