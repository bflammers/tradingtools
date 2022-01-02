from decimal import Decimal
from typing import Dict
from random import uniform

from tradingtools.utils import split_pair

from .strategy import AbstractStrategy, StrategyConfig
from ..assets import PortfolioAsset
from ..data import AbstractData
from ..utils import float_to_decimal


class DummyStrategy(AbstractStrategy):
    def optimal_proportions(
        self, data: AbstractData, portfolio: PortfolioAsset
    ) -> Dict[str, Decimal]:

        asset_names = portfolio.get_asset_names()

        propensities = [uniform(0.0, 10.0) for _ in asset_names]
        total_propensity = sum(propensities)

        proportions = {
            base: float_to_decimal(propensity / total_propensity)
            for base, propensity in zip(asset_names, propensities)
        }

        return proportions
