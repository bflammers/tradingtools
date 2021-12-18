

class AbstractAssetVisitor:

    def __init__(self) -> None:
        pass

    def visit_composite_asset(self, asset) -> None:
        raise NotImplementedError

    def visit_symbol_asset(self, asset) -> None:
        raise NotImplementedError

    def leave(self) -> None:
        raise NotImplementedError

