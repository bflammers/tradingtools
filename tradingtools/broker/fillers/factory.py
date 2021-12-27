from .filler import AbstractFillStrategy, FillStrategyConfig, FillerTypes
from .marketorder import MarketOrderFillStrategy

from ..exchanges import AbstractExchange
from ...assets import AbstractCompositeAsset, SymbolAsset


def filler_factory(
    base_asset: AbstractCompositeAsset,
    quote_asset: AbstractCompositeAsset,
    exchange: AbstractExchange,
    config: FillStrategyConfig,
) -> AbstractFillStrategy:

    if config.type is FillerTypes.marketorder:
        return MarketOrderFillStrategy(
            base_asset=base_asset,
            quote_asset=quote_asset,
            exchange=exchange,
            config=config,
        )
    else:
        raise NotImplementedError(
            f"[filler_factory] FillStrategy with type {config.type} not implemented"
        )
