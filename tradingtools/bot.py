import asyncio

from dataclasses import dataclass
from decimal import Decimal
from typing import List
from logging import getLogger

from .assets import CompositeAsset, SymbolAsset
from .broker import Broker, BrokerConfig
from .strategies import AbstractStrategy, strategy_factory
from .data import AbstractData, AbstractDataLoader, dataloader_factory
from .visitors import AbstractAssetVisitor, visitor_factory


logger = getLogger(__name__)


@dataclass
class BotConfig:
    strategy__config: dict
    data_loader__config: dict
    visitors__config: List[dict]
    broker__config: BrokerConfig
    backtest: bool = True
    assets__time_diff_tol_sec: float = 180.0

    def __post_init__(self) -> None:

        # Ensure backtest is set to the same value everywhere
        if self.backtest != self.broker__config.backtest:
            logger.error(
                f"[BotConfig] backtest in broker_config = {self.broker__config.backtest}"
            )


class Bot:

    _config: BotConfig
    _portfolio: CompositeAsset
    _strategy: AbstractStrategy
    _data_loader: AbstractDataLoader
    _visitors: List[AbstractAssetVisitor]

    def __init__(self, config: BotConfig) -> None:

        self._config = config
        self._portfolio = CompositeAsset(
            "portfolio", self._config.assets__time_diff_tol_sec
        )
        self._broker = Broker(self._config.broker__config)

        # Factories
        self._strategy = strategy_factory(self._config.strategy__config)
        self._data_loader = dataloader_factory(self._config.data_loader__config)
        self._visitors = visitor_factory(self._config.visitors__config)

        # TODO: still need a sync method to synchronize assets with exchange state

    def start(self, loop: asyncio.AbstractEventLoop) -> None:

        loop.create_task(self.run())

    async def run(self) -> None:

        asset = SymbolAsset("EUR/EUR")
        asset.set_price(Decimal("1"))
        asset.set_quantity(Decimal("1000"))
        self._portfolio.add_asset(asset)

        self._create_assets()

        async for data in self._data_loader.load():

            # Update prices
            self._portfolio.update_prices(data.get_latest())

            # Apply visitors
            self._visit_portfolio()

            # Evaluate strategy, results in the gaps that need to be filled on
            # different markets to top up to assets to the uptimal quantities
            market_gaps = self._strategy.evaluate(data, self._portfolio)

            # Make orders to fill market gaps in portfolio
            await self._broker.fill(market_gaps, self._portfolio)

    def _visit_portfolio(self) -> None:
        for visitor in self._visitors:
            self._portfolio.accept(visitor)
            visitor.leave()

    def _create_assets(self) -> None:

        # TODO: make this create assets for every asset, not for every pair
        for pair in self._data_loader.get_pairs():

            if not self._portfolio.get_asset(pair):
                logger.info(f"[Bot] creating asset {pair}")
                asset = SymbolAsset(pair, self._config.assets__time_diff_tol_sec)
                self._portfolio.add_asset(asset)
