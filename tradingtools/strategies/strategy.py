from dataclasses import dataclass
from decimal import Decimal
from enum import Enum
from typing import Dict

from ..assets import PortfolioAsset
from ..data import AbstractData


class StrategyTypes(Enum):
    dummy = "dummy"


@dataclass
class StrategyConfig:
    type: StrategyTypes
    smooth_value_base_thr: Decimal = Decimal("10.0")
    smooth_value_diff_thr: Decimal = Decimal("5.0")


class AbstractStrategy:
    _config: StrategyConfig

    def __init__(self, config: StrategyConfig) -> None:
        self._config = config

    def evaluate(
        self, data: AbstractData, assets: PortfolioAsset
    ) -> Dict[str, Decimal]:

        # Determin optimal proportions
        proportions = self.optimal_proportions(data, assets)
        self._check_proportions(proportions)

        # Convert to quantities
        quantities = self._to_quantities(proportions, assets)


        market_gaps = self._to_market_gaps(quantities)

        # Smooth quantities signal to avoid too many jumps
        smooth_gaps = self.smooth_quantities(market_gaps, assets)

        # TODO: make this output a market gaps dict
        return smooth_quantities

    def optimal_proportions(
        self, data: AbstractData, assets: PortfolioAsset
    ) -> Dict[str, Decimal]:
        raise NotImplementedError

    def smooth_quantities(
        self, quantities: Dict[str, Decimal], assets: PortfolioAsset = None
    ) -> Dict[str, Decimal]:

        for pair, quantity in quantities.items():

            asset = assets.get_asset(pair)

            # Avoiding lots of small assets - check if value diff from zero
            # is higher than threshold. If not, smooth to zero
            _, value_diff = asset.get_difference(Decimal("0.0"))
            if abs(value_diff) < self._config.smooth_value_base_thr:
                quantities[pair] = Decimal("0.0")

            # Avoiding many small orders - check if value diff from current
            # is higher than threshold. If not, smooth to current quantity
            _, value_diff = asset.get_difference(quantity)
            if abs(value_diff) < self._config.smooth_value_diff_thr:
                quantities[pair] = asset.get_quantity()

        return quantities

    @staticmethod
    def _check_proportions(proportions: Dict[str, Decimal]) -> None:

        total = 0
        for pair, proportion in proportions.items():

            if proportion < Decimal("0") or proportion > Decimal("1"):
                raise ValueError(
                    f"[Strategy] proportion of {proportion} for {pair} not >= 0 and <= 1"
                )

            total += proportion

        if total < Decimal("0") or total > Decimal("1"):
            raise ValueError(
                f"[Strategy] total of proportions is {total}, not >= 0 and <= 1"
            )

    @staticmethod
    def _to_quantities(
        proportions: Dict[str, Decimal], assets: PortfolioAsset
    ) -> Dict[str, Decimal]:

        total_value = assets.get_value()

        quantities = {}
        for pair, proportion in proportions.items():

            # Determine value that should be allocated in this asset
            share_value = total_value * proportion

            # Determine corresponding quantity
            price = assets.get_asset(pair).get_price()
            quantity = share_value / price

            quantities[pair] = quantity

        return quantities
