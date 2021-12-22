from typing import List

from .visitor import AbstractAssetVisitor, AssetVisitorConfig, AssetVisitorTypes


def _visitor_factory(config: AssetVisitorConfig) -> AbstractAssetVisitor:
    raise NotImplementedError

def visitor_factory(configs: List[AssetVisitorConfig]) -> List[AbstractAssetVisitor]:
    return [_visitor_factory(config) for config in configs]


