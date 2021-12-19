from .filler import AbstractFillStrategy, FillStrategyConfig
from .marketorder import MarketOrderFillStrategy

from ..exchanges import AbstractExchange
from ...assets import SymbolAsset


def filler_factory(
    asset: SymbolAsset, exchange: AbstractExchange, config: FillStrategyConfig
) -> AbstractFillStrategy:

    if config.type == "market":
        return MarketOrderFillStrategy(asset, exchange, config)
    else:
        raise NotImplementedError(
            f"[filler_factory] FillStrategy with type {config.type} not implemented"
        )
