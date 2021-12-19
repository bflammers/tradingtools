import asyncio

from dataclasses import dataclass
from typing import List
from logging import getLogger

from .assets import Prices, CompositeAsset
from .broker import AbstractBroker, BrokerConfig, broker_factory
from .strategies import AbstractStrategy, strategy_factory
from .data import AbstractDataLoader, dataloader_factory
from .visitors import AbstractAssetVisitor, visitor_factory


logger = getLogger(__name__)


@dataclass
class BotConfig:
    strategy__config: dict
    data_loader__config: dict
    visitors__config: List[dict]
    broker__config: BrokerConfig
    backtest: bool = True
    prices__time_tol_sec: float = 180.0

    def __post_init__(self) -> None:

        # Ensure backtest is set to the same value everywhere
        if self.backtest != self.broker__config.backtest:
            logger.error(
                f"[BotConfig] backtest in broker_config = {self.broker__config.backtest}"
            )


class Bot:

    _config: BotConfig
    _prices: Prices
    _portfolio: CompositeAsset
    _strategy: AbstractStrategy
    _data_loader: AbstractDataLoader
    _visitors: List[AbstractAssetVisitor]

    def __init__(self, config: BotConfig) -> None:

        self._config = config
        self._prices = Prices(time_tol_sec=self._config.prices__time_tol_sec)
        self._portfolio = CompositeAsset("portfolio", self._prices)

        # Factories
        self._broker = broker_factory(self._config.broker__config)
        self._strategy = strategy_factory[self._config.strategy__config]
        self._data_loader = dataloader_factory[self._config.data_loader__config]
        self._visitors = visitor_factory(self._config.visitors__config)

    async def run(self) -> None:

        async for data in self._data_loader():

            # Update prices
            self._update_prices(data)

            # Apply visitors
            self._visit_portfolio()

            # Evaluate strategy, results in optimal quantities
            opt_quantities = self._strategy.evaluate(data, self._portfolio)

            # Make orders to fill portfolio up to optimal positions
            await self._broker.fill(self._portfolio, opt_quantities)

    def _visit_portfolio(self) -> None:
        for visitor in self._visitors:
            self._portfolio.accept(visitor)
            visitor.leave()

    def _update_prices(self, data) -> None:
        latest_prices = data.get_latest()
        for (
            symbol,
            price,
        ) in latest_prices:
            self._prices.update(symbol, price)
