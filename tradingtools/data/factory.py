from tradingtools.data.dummy import DummyDataLoader
from .dataloader import AbstractDataLoader, DataLoaderConfig, DataLoaderTypes


def dataloader_factory(config: DataLoaderConfig) -> AbstractDataLoader:
    
    if config.type is DataLoaderTypes.dummy:
        return DummyDataLoader(config)
    else:
        raise NotImplementedError(
            f"[dataloader_factory] DataLoader type {config.type} not supported"
        )
