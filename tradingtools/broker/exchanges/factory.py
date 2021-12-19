
from .exchange import AbstractExchange, ExchangeConfig
from .binance import BinanceExchange
from .dummy import DummyExchange

def exchange_factory(config: ExchangeConfig) -> AbstractExchange:

    if config.type == 'binance':
        return BinanceExchange(config)
    elif config.type == 'dummy':
        return DummyExchange(config)
    else:
        raise NotImplementedError(
            f"[exchange_factory] Exchange type {config.type} not supported"
        )