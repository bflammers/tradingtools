import csv

from decimal import Decimal
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path

from .visitor import AssetVisitorConfig, AbstractAssetVisitor


@dataclass
class LogAssetVisitorConfig(AssetVisitorConfig):
    results_dir: str = f"./results"


class LogAssetVisitor(AbstractAssetVisitor):
    def __init__(self, config: LogAssetVisitorConfig) -> None:
        super().__init__(config)
        self._q = []
        self._columns = ["datetime", "name", "quantity", "price", "value"]
        self._path = (
            Path(self._config.results_dir)
            / f"{datetime.now().strftime('%Y-%m-%d %H%M%S')}.csv"
        )
        self._create_csv()

    def _create_csv(self) -> None:

        # Create parent dir if not exists
        self._path.parent.mkdir(parents=True, exist_ok=True)

        # Create file and write header
        with open(self._path, "w") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(self._columns)

    def _add_entry(
        self, name: str, quantity: Decimal, price: Decimal, value: Decimal
    ) -> None:
        self._q.append(
            {
                "datetime": datetime.now().isoformat(),
                "name": name,
                "quantity": quantity,
                "price": price,
                "value": value,
            }
        )

    def visit_composite_asset(self, asset) -> None:
        self._add_entry(
            name=asset.get_name(),
            quantity=asset.get_quantity(),
            price=asset.get_price(),
            value=asset.get_value(),
        )

    def visit_symbol_asset(self, asset) -> None:
        self._add_entry(
            name=asset.get_name(),
            quantity=asset.get_quantity(),
            price=asset.get_price(),
            value=asset.get_value(),
        )

    def leave(self) -> None:

        if self._q:
            with open(self._path, "a") as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=self._columns)
                writer.writerows(self._q)

        self._q = []
