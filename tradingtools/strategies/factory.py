from .strategy import AbstractStrategy, StrategyConfig, StrategyTypes
from .dummy import DummyStrategy


def strategy_factory(config: StrategyConfig) -> AbstractStrategy:

    if config.type is StrategyTypes.dummy:
        return DummyStrategy(config)
    else:
        raise NotImplementedError(
            f"[strategy_factory] Strategy with type {config.type} not implemented"
        )
