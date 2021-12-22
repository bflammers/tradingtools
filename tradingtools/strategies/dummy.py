from decimal import Decimal
from typing import Dict
from random import uniform

from .strategy import AbstractStrategy, StrategyConfig
from ..assets import CompositeAsset
from ..data import AbstractData


class DummyStrategy(AbstractStrategy):
    def optimal_proportions(
        self, data: AbstractData, assets: CompositeAsset
    ) -> Dict[str, Decimal]:

        pairs = data.get_pairs()

        propensities = [uniform(0.0, 10.0) for _ in pairs]
        total_propensity = sum(propensities) * uniform(1.0, 5.0)

        proportions = {
            pair: Decimal(propensity / total_propensity)
            for pair, propensity in zip(pairs, propensities)
        }

        return proportions
