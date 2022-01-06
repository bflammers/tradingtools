import asyncio

from dataclasses import dataclass
from decimal import Decimal
from typing import List
from logging import getLogger

from tradingtools.utils import RunType, split_pair

from .assets import PortfolioAsset, SymbolAsset
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
    run_type: RunType = RunType.backtest
    default_quote_symbol: str = "USDT"
    default_quote_starting_capital: Decimal = Decimal("1000")
    assets__time_diff_tol_sec: float = 180.0

    def __post_init__(self) -> None:

        # Ensure backtest is set to the same value everywhere
        logger.info(
            f"[BotConfig] setting broker__config.run_type to {self.broker__config.run_type}"
        )
        self.broker__config.run_type = self.run_type


class Bot:
    def __init__(self, config: BotConfig) -> None:

        self._config: BotConfig = config
        self._portfolio: PortfolioAsset = PortfolioAsset(
            name="portfolio",
            default_quote=self._config.default_quote_symbol,
            time_diff_tol_sec=self._config.assets__time_diff_tol_sec,
        )
        self._broker = Broker(self._config.broker__config)

        # Factories
        self._strategy: AbstractStrategy = strategy_factory(
            self._config.strategy__config
        )
        self._data_loader: AbstractDataLoader = dataloader_factory(
            self._config.data_loader__config
        )
        self._visitors: List[AbstractAssetVisitor] = visitor_factory(
            self._config.visitors__config
        )

    def start(self, loop: asyncio.AbstractEventLoop) -> None:

        loop.create_task(self.run())

    async def run(self) -> None:

        self._create_assets()

        async for data in self._data_loader.load_async():

            # Update prices
            prices = data.get_latest()
            self._portfolio.update_prices(prices)
            
            # Log portfolio
            logger.info(f"\n[Bot] portfolio: {self._portfolio}")

            # Apply visitors
            self._visit_portfolio()

            # Evaluate strategy, results in the gaps that need to be filled on
            # different markets to top up to assets to the uptimal quantities
            market_gaps = self._strategy.evaluate(data, self._portfolio)
            logger.debug(f"[Bot] market gaps: {market_gaps}")

            # Make orders to fill market gaps in portfolio
            await self._broker.fill(market_gaps, self._portfolio)


    def _visit_portfolio(self) -> None:
        for visitor in self._visitors:
            self._portfolio.accept(visitor)
            visitor.leave()

    def _create_assets(self) -> None:

        # Add asset for default quote symbol
        if self._config.default_quote_symbol:
            self._create_default_quote_asset()

        # Add all base asset present in dataloader
        self._create_remaining_assets()

    def _create_remaining_assets(self):

        # Create a set of all (base AND quote) asset names
        pairs = set()
        for pair in self._data_loader.get_pairs():
            base, quote = split_pair(pair)
            pairs |= {base, quote}

        # Add assets for all base symbols
        for name in pairs:

            if not self._portfolio.get_asset(name):
                logger.info(f"[Bot] creating asset {name}")
                asset = SymbolAsset(
                    name=name,
                    default_quote=self._config.default_quote_symbol,
                    time_diff_tol_sec=self._config.assets__time_diff_tol_sec,
                )
                self._portfolio.add_asset(asset)

    def _create_default_quote_asset(self):

        logger.info(
            f"[Bot] creating default quote asset {self._config.default_quote_symbol}"
        )
        asset = SymbolAsset(
            name=self._config.default_quote_symbol,
            default_quote=self._config.default_quote_symbol,
            time_diff_tol_sec=None,
        )

        # Defined as unit measure
        asset.set_price(Decimal("1.0"))

        # If backtest, add starting capital
        if self._config.run_type is RunType.backtest:
            asset.set_quantity(self._config.default_quote_starting_capital)

        self._portfolio.add_asset(asset)
