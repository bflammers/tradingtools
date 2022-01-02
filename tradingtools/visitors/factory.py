from typing import List

from .visitor import AbstractAssetVisitor, AssetVisitorConfig, AssetVisitorTypes
from .logging import LogAssetVisitor


def _visitor_factory(config: AssetVisitorConfig) -> AbstractAssetVisitor:
    if config.type is AssetVisitorTypes.logger:
        return LogAssetVisitor(config)

    raise NotImplementedError(
        f"[visitor_factory] visitor type {config.type} not implemented"
    )


def visitor_factory(configs: List[AssetVisitorConfig]) -> List[AbstractAssetVisitor]:
    return [_visitor_factory(config) for config in configs]
