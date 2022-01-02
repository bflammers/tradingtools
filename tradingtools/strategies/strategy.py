from dataclasses import dataclass
from decimal import Decimal
from enum import Enum
from typing import Dict
from logging import getLogger

from tradingtools.utils import split_pair

from ..assets import PortfolioAsset
from ..data import AbstractData
from ..utils import round_decimal


logger = getLogger(__name__)


class StrategyTypes(Enum):
    dummy = "dummy"


@dataclass
class StrategyConfig:
    type: StrategyTypes
    smooth_value_base_thr: Decimal = Decimal("10.0")
    smooth_value_diff_thr: Decimal = Decimal("5.0")


class AbstractStrategy:
    def __init__(self, config: StrategyConfig) -> None:
        self._config: StrategyConfig = config

    def evaluate(
        self, data: AbstractData, portfolio: PortfolioAsset
    ) -> Dict[str, Decimal]:

        # Determine optimal proportions
        proportions = self.optimal_proportions(data, portfolio)
        self._check_proportions(proportions)

        # Pick which markets to buy/sell the assets on
        market_gaps = self.market_gaps(proportions, portfolio)

        return market_gaps

    def optimal_proportions(
        self, data: AbstractData, portfolio: PortfolioAsset
    ) -> Dict[str, Decimal]:
        raise NotImplementedError

    @staticmethod
    def _check_proportions(proportions: Dict[str, Decimal]) -> None:

        total = Decimal("0")
        for name, proportion in proportions.items():

            if proportion < Decimal("0") or proportion > Decimal("1"):
                logger.error(
                    f"[Strategy] proportion of {proportion} for {name} not >= 0 and <= 1"
                )

            total += proportion

        # Round to avoid a total > 1, due to float representation
        total = round_decimal(total, 4)

        if total < Decimal("0") or total > Decimal("1"):
            raise ValueError(
                f"[Strategy] total of proportions is {total}, not >= 0 and <= 1"
            )

    def _smooth_gaps(
        self, gaps: Dict[str, Decimal], portfolio: PortfolioAsset = None
    ) -> Dict[str, Decimal]:

        for pair, gap in gaps.items():

            base, quote = split_pair(pair)
            asset = portfolio.get_asset(base)

            # Round to avoid gaps with a very large number of decimal places
            gaps[pair] = round_decimal(gap)

            # Avoiding lots of small assets - check if value diff from zero
            # is higher than threshold. If not, smooth to zero
            value = asset.get_price(quote) * gap
            if abs(value) < self._config.smooth_value_base_thr:
                gaps[pair] = Decimal("0.0")

            # Avoiding many small orders - check if value diff from current
            # is higher than threshold. If not, smooth to current quantity
            value_diff = asset.get_value_difference(gap, quote)
            if abs(value_diff) < self._config.smooth_value_diff_thr:
                gaps[pair] = Decimal("0.0")

        return gaps

    def market_gaps(
        self, proportions: Dict[str, Decimal], portfolio: PortfolioAsset
    ) -> Dict[str, Decimal]:

        total_value = portfolio.get_value()

        gaps = {}
        for base, proportion in proportions.items():

            # Get asset, determine market
            asset = portfolio.get_asset(base)
            market = asset.get_market()
            _, quote = split_pair(market)

            # Don't generate an order when default quote
            if base == quote:
                continue

            # Determine value to allocated in this asset, corresponding quantity
            share_value = total_value * proportion
            target_quantity = share_value / asset.get_price(quote)

            # Determine quantity difference, add to dict
            quantity_diff = target_quantity - asset.get_quantity()
            gaps[market] = quantity_diff

        # Smooth quantities to avoid many small jumps
        gaps = self._smooth_gaps(gaps, portfolio)

        return gaps
