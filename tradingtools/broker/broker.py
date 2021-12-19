import asyncio
from asyncio.futures import Future

from logging import getLogger
from typing import Dict
from decimal import Decimal
from dataclasses import dataclass

from tradingtools.broker.exchanges import exchange

from ..assets import AbstractAsset, CompositeAsset
from ..utils import Order
from .fillers import AbstractFillStrategy, FillStrategyConfig, filler_factory
from .exchanges import AbstractExchange, ExchangeConfig, exchange_factory


logger = getLogger(__name__)


@dataclass
class BrokerConfig:
    type: str
    backtest: bool
    filler__config: FillStrategyConfig
    exchange__config: ExchangeConfig


class Broker:

    _exchange: AbstractExchange
    _fillers: Dict[str, AbstractFillStrategy] = []

    def __init__(self, config: BrokerConfig) -> None:
        self._config = config

        # Factories
        self._exchange = exchange_factory(self._config.exchange__config)

    def _get_or_create_filler(self, asset: AbstractAsset):

        asset_name = asset.get_name()

        if asset_name not in self._fillers:
            self._fillers[asset_name] = filler_factory(
                asset, self._exchange, self._config.filler__config
            )

        return self._fillers[asset_name]

    async def fill(
        self, portfolio: CompositeAsset, opt_quantities: Dict[str, Decimal]
    ) -> Future:

        fill_coroutines = []

        for name, opt_quantity in opt_quantities.items():

            # Get or create asset and filler for asset if it does not exist yet
            asset = portfolio.get_or_create_symbol_asset(name)
            filler = self._get_or_create_filler(asset)

            # Append to list of coroutines 
            fill_coroutines.append(filler.fill(opt_quantity))

        return asyncio.gather(*fill_coroutines)
