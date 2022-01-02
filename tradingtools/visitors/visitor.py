from dataclasses import dataclass
from enum import Enum


class AssetVisitorTypes(Enum):
    dummy = "dummy"
    logger = "logger"


@dataclass
class AssetVisitorConfig:
    type: AssetVisitorTypes


class AbstractAssetVisitor:

    def __init__(self, config: AssetVisitorConfig) -> None:
        self._config: AssetVisitorConfig = config

    def visit_composite_asset(self, asset) -> None:
        raise NotImplementedError

    def visit_symbol_asset(self, asset) -> None:
        raise NotImplementedError

    def leave(self) -> None:
        raise NotImplementedError
