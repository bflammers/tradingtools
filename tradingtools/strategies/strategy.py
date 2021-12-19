
from ..assets import CompositeAsset

class AbstractStrategy:

    def evaluate(self, data, assets: CompositeAsset) -> None:
        pass

