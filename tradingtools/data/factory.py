from tradingtools.data.dummy import DummyDataLoader
from tradingtools.data.historical import HistoricalDataLoader
from .dataloader import AbstractDataLoader, DataLoaderConfig, DataLoaderTypes


def dataloader_factory(config: DataLoaderConfig) -> AbstractDataLoader:
    
    if config.type is DataLoaderTypes.dummy:
        return DummyDataLoader(config)
    elif config.type is DataLoaderTypes.historical:
        return HistoricalDataLoader(config)
    else:
        raise NotImplementedError(
            f"[dataloader_factory] DataLoader type {config.type} not supported"
        )
