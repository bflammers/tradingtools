
from dataclasses import dataclass

from typing import List

from .assets import Prices, CompositeAsset
from .broker import AbstractBroker
from .strategies import AbstractStrategy, strategy_factory
from .data import AbstractDataLoader, dataloader_factory
from .visitors import AbstractAssetVisitor, visitor_factory


@dataclass
class BotConfig:
    strategy__config: dict
    data_loader__config: dict
    visitors__config: List[dict]
    backtest: bool = True
    prices__time_tol_sec: float = 180.0


class Bot:

    _config: BotConfig
    _prices: Prices 
    _portfolio: CompositeAsset
    _strategy: AbstractStrategy
    _data_loader: AbstractDataLoader
    _visitors: List[AbstractAssetVisitor]

    def __init__(self, config: BotConfig, exchange_credentials: dict) -> None:
        
        self._config = config
        self._prices = Prices(time_tol_sec=self._config.prices__time_tol_sec)
        self._portfolio = CompositeAsset("portfolio", self._prices)
        self._broker = AbstractBroker(exchange_credentials, self._config.backtest)
        self._strategy = strategy_factory[self._config.strategy__config]
        self._data_loader = dataloader_factory[self._config.data_loader__config]
        self._visitors = visitor_factory(self._config.visitors__config)

    async def run(self) -> None:
        
        async for data in self._data_loader():
            
            # Update prices
            self._update_prices(data)

            # Apply visitors
            self._visit_portfolio()

            # Evaluate strategy
            deltas = self._strategy.evaluate(data, self._portfolio)

            # Make orders to close the deltas
            await self._broker.fill(deltas)

    def _visit_portfolio(self) -> None:
        for visitor in self._visitors:
            self._portfolio.accept(visitor)
            visitor.leave()

    def _update_prices(self, data) -> None:
        latest_prices = data.get_latest()
        for symbol, price, in latest_prices:
            self._prices.update(symbol, price)