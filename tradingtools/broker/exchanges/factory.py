
from .exchange import AbstractExchange, ExchangeConfig, ExchangeTypes
from .binance import BinanceExchange
from .dummy import DummyExchange

def exchange_factory(config: ExchangeConfig) -> AbstractExchange:

    if config.type is ExchangeTypes.binance:
        return BinanceExchange(config)
    elif config.type is ExchangeTypes.dummy:
        return DummyExchange(config)
    else:
        raise NotImplementedError(
            f"[exchange_factory] Exchange type {config.type} not supported"
        )
        