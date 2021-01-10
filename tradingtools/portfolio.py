import pandas as pd
import numpy as np
import uuid
import csv

from uuid import uuid4
from decimal import Decimal
from pathlib import Path


try:
    from .utils import (
        colors,
        extract_prices,
        warnings,
        timestamp_to_string,
        print_item,
    )
    from .symbol import Symbol
except:
    from utils import (
        colors,
        extract_prices,
        warnings,
        timestamp_to_string,
        print_item,
    )
    from symbol import Symbol


class Portfolio:
    def __init__(
        self,
        start_capital: Decimal = None,
        results_parent_dir: str = "./runs",
        reference_currency: str = "EUR",
        verbose: bool = True,
    ) -> None:
        super().__init__()

        self._verbose = verbose
        self._reference_currency = reference_currency
        self.symbols = {}

        # Create directory for results
        now = pd.Timestamp.now().strftime("%Y%m%d_%H%M")
        ts_uuid = f"{now}_{uuid4().hex}"
        self._results_dir = (Path(results_parent_dir) / ts_uuid).absolute()
        self._results_dir.mkdir(parents=True, exist_ok=True)

        if self._verbose:
            print(f"[Portfolio] results directory created: {self._results_dir}")

        # Create csv file and path for ticks
        self._tick_ohlcv_path = self._results_dir / f"{now}_ticks.csv"
        self._tick_ohlcv_columns = [
            "tick_id",
            "trading_pair",
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "volume",
        ]
        self._create_csv(self._tick_ohlcv_path, self._tick_ohlcv_columns)

        # Create csv file and path for optimal_positions
        self._opt_positions_path = self._results_dir / f"{now}_optimal_positions.csv"
        self._opt_positions_columns = ["positions_id", "timestamp", "symbol", "amount"]
        self._create_csv(self._opt_positions_path, self._opt_positions_columns)

        # Create csv file and path for orders
        self._orders_path = self._results_dir / f"{now}_orders.csv"
        self._orders_columns = [
            "order_id",
            "trading_pair",
            "side",
            "amount",
            "timestamp_tick",
            "price_execution",
            "cost_execution",
            "timestamp_execution",
        ]
        self._create_csv(self._orders_path, self._orders_columns)

        # Create csv file and path for orders
        self._settlements_path = self._results_dir / f"{now}_settlements.csv"
        self._settlements_columns = [
            "order_id",
            "trading_pair",
            "side",
            "amount",
            "timestamp_tick",
            "price_execution",
            "cost_execution",
            "timestamp_execution",
            "price_settlement",
            "timestamp_settlement",
            "fee",
            "fee_currency",
            "fee_reference_currency",
            "slippage",
            "order_value",
        ]
        self._create_csv(self._settlements_path, self._settlements_columns)

        # Initialize state objects
        self._current_optimal_positions = {}
        self._reserved_capital = Decimal(0)

        self._start_capital = start_capital
        self._unallocated_capital = start_capital

    @staticmethod
    def _create_csv(path: str, header: list) -> None:

        # Create file and write header
        with open(path, "w") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(header)

    def _append_to_csv(self, target_csv: str, new_values: dict) -> None:

        if target_csv == "tick_ohlcv":
            column_names = self._tick_ohlcv_columns
            csv_path = self._tick_ohlcv_path
        elif target_csv == "opt_positions":
            column_names = self._opt_positions_columns
            csv_path = self._opt_positions_path
        elif target_csv == "orders":
            column_names = self._orders_columns
            csv_path = self._orders_path
        elif target_csv == "settlements":
            column_names = self._settlements_columns
            csv_path = self._settlements_path
        else:
            raise NotImplementedError("[Portfolio] no append method for {target_csv}")

        row = []

        for column in column_names:

            try:
                row.append(new_values[column])
            except KeyError:
                row.append(None)
                warnings.warn(
                    f"[Portfolio._append_to_csv] key-value pair for {column} not in new values for {target_csv}"
                )

        with open(csv_path, "a") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(row)

    def update(self, tick: list, optimal_positions: dict = {}) -> list:

        """Update the portfolio using the new optimal positions (from strategy)
        and most recent prices. The optimal positions only trigger an order request
        to the broker in case the optimal position received from the strategy is different
        from the current position. The value of the assets are updated on each call to this
        method using the new prices. The asset volumes are only updated through a call to the
        add_settlement method when a position has been filled.

        Returns:
            list: list of order requests that will be passed to the broker
        """

        # optimal_positions = [
        #     {"symbol": "BTCUSD", "volume": 10},
        #     {"symbol": "ETHUSD", "volume": 0},
        # ]

        if self._start_capital is None:
            raise Exception("No starting capital --> first call initialize")

        # Write tick to file
        self._write_tick_ohlcv(tick)

        # Construct orders list to append orders to
        orders = []
        
        # Sync prices
        self.sync_prices(tick)

        # Update positions
        for trading_pair, amount in optimal_positions.items():

            base, quote = trading_pair.split("/")

            # Add symbol if needed
            if base not in self.symbols:
                self.symbols[base] = Symbol(base)
                self._current_optimal_positions[base] = 0

            # Load symbol
            symbol = self.symbols[base]

            # # Update price and tick timestamp
            # price, timestamp = self._extract_price_from_tick(tick, trading_pair)
            # symbol.sync_state(tick_timestamp=timestamp, price=price)

            if amount != symbol.optimal_amount:

                # Update symbol position, generate order if needed
                order = symbol.update_optimal_position(amount)

                if order is not None:

                    # Print order
                    if self._verbose:
                        print(f"[Portfolio.update] new order: {order}")

                    # Append orders if not None
                    orders.append(order)

                    # Reserve capital if buy
                    if order["side"] == "buy":
                        self._reserved_capital += order["cost_execution"]

        # Write optimal postions if changed
        if self._current_optimal_positions != optimal_positions:
            self._write_optimal_positions(optimal_positions)

        # Update optimal positions
        self._current_optimal_positions = optimal_positions

        # Write order to log if any
        if orders:
            self._write_orders(orders)

        return orders
        
    def initialize(self, symbol_amounts: dict, tick: list) -> None:
        
        self.sync_amounts(symbol_amounts)
        self.sync_prices(tick)
        
        starting_capital = 0
        
        for symbol in self.symbols.values():

            starting_capital += symbol.get_current_value()
        
        self._unallocated_capital = symbol_amounts[self._reference_currency]
        self._start_capital = starting_capital + self._unallocated_capital
                
        
    def sync_prices(self, tick: list) -> None:
        
        for symbol_name, symbol in self.symbols.items():

            trading_pair = f"{symbol_name}/{self._reference_currency}"
            price, timestamp = self._extract_price_from_tick(tick, trading_pair)

            symbol.sync_state(tick_timestamp=timestamp, price=price)

    def sync_amounts(self, symbol_amounts: dict) -> None:

        for symbol_name, amount in symbol_amounts.items():

            if symbol_name != self._reference_currency:

                trading_pair = f"{symbol_name}/{self._reference_currency}"
                amount = symbol_amounts[symbol_name]

                if symbol_name not in self.symbols:
                    self.symbols[symbol_name] = Symbol(symbol_name)

                symbol = self.symbols[symbol_name]
                symbol.sync_state(current_amount=amount)

    @staticmethod
    def _extract_price_from_tick(
        tick: list, trading_pair: str, price_type: str = "close"
    ) -> tuple:
        for t in tick:
            if t["trading_pair"] == trading_pair:
                return t[price_type], t["timestamp"]

    def _write_tick_ohlcv(self, tick: list) -> None:

        # Fields common for all symbols within this position
        tick_id = uuid.uuid4().hex

        # Update volume for each symbol, add new if not yet present
        for new_row in tick:

            # Add tick id
            new_row["tick_id"] = tick_id

            # Append new row
            self._append_to_csv("tick_ohlcv", new_row)

    def _write_optimal_positions(self, new_positions: dict) -> None:

        """Internal method to update the intended positions. Note that this method
        only updates the "ideal" positions, and not the actual orders that will be
        created for the broker (these will be created through the _construct_orders method)
        """

        # Fields common for all symbols within this position
        positions_id = uuid.uuid4().hex
        timestamp = timestamp_to_string(pd.Timestamp.now())

        # Update volume for each symbol, add new if not yet present
        for symbol_name, amount in new_positions.items():

            new_row = {
                "positions_id": positions_id,
                "timestamp": timestamp,
                "symbol": symbol_name,
                "amount": amount,
            }

            # Append new row
            self._append_to_csv("opt_positions", new_row)

    def _write_orders(self, orders: list) -> None:

        for order in orders:
            # Append to csv file
            self._append_to_csv("orders", order)

    def settle_order(
        self,
        trading_pair: str,
        order_id: str,
        order_value: Decimal,
        price_settlement: Decimal,
        timestamp_settlement: pd.Timestamp,
        fee: Decimal,
        fee_currency: str,
    ) -> None:

        """Adds a settlement to an order and updates asset volumes and value at buy when the
        order has been filled. Uses the order_id to id the corresponding order in the orders df
        and updates with price (settlement) and timestamp (settlement)
        """

        # Split trading pair
        base, quote = trading_pair.split("/")

        # Get symbol
        symbol = self.symbols[base]

        # Settle order
        settled_order = symbol.add_settlement(
            order_id, order_value=Decimal(order_value)
        )

        # Add settlement details to order
        settled_order["price_settlement"] = Decimal(price_settlement)
        settled_order["timestamp_settlement"] = timestamp_settlement
        settled_order["fee"] = Decimal(fee)
        settled_order["fee_currency"] = fee_currency
        settled_order["slippage"] = Decimal(price_settlement) - Decimal(
            settled_order["price_execution"]
        )
        settled_order["order_value"] = Decimal(order_value)

        # Convert fee to base currency if needed
        if fee_currency != self._reference_currency:
            fee_reference_currency = Decimal(price_settlement) * Decimal(fee)
            settled_order["fee_reference_currency"] = fee_reference_currency
        else:
            settled_order["fee_reference_currency"] = fee

        # Update unallocated and reserved capital
        if settled_order["side"] == "buy":
            # Free reserved capital only if side was buy
            self._reserved_capital -= order_value

            # Subtract from unallocated capital
            self._unallocated_capital -= order_value

        elif settled_order["side"] == "sell":
            # Add to unallocated capital
            self._unallocated_capital += order_value

        else:
            raise Exception("[Portfolio.settle_order] side order not buy or sell")

        # Write to csv
        self._append_to_csv("settlements", settled_order)

    def get_optimal_positions(self) -> pd.DataFrame:
        df = pd.read_csv(self._opt_positions_path)
        return df

    def get_orders(self) -> pd.DataFrame:
        df = pd.read_csv(self._orders_path)
        return df

    def get_settled_orders(self) -> pd.DataFrame:
        df = pd.read_csv(self._settlements_path)
        return df

    def profit_and_loss(self) -> dict:

        """Method for retrieving the current profit and loss with the latest prices

        Returns:
            dict: overview of current pnl, overall and per asset
        """

        # Get pnl of all symbols
        pnl_symbols = {}
        current_value = self._unallocated_capital
        n_orders = 0
        n_open_orders = 0

        for symbol_name, symbol in self.symbols.items():

            pnl_symbol = symbol.profit_and_loss()
            pnl_symbols[symbol_name] = pnl_symbol

            current_value += pnl_symbol["value"]
            n_orders += pnl_symbol["n_orders"]
            n_open_orders += pnl_symbol["n_open_orders"]

        # Calculate profit percentage total
        profit_percentage = (
            (current_value - self._start_capital) / self._start_capital * 100
        )

        pnl = {
            "start_capital": self._start_capital,
            "unallocated": self._unallocated_capital,
            "reserved": self._reserved_capital,
            "total_value": current_value,
            "profit_percentage": profit_percentage,
            "n_orders": n_orders,
            "n_open_orders": n_open_orders,
            "symbols": pnl_symbols,
        }

        return pnl

    def __str__(self) -> str:

        # Total: xxx / +xx% / x orders -- Unallocated: xxx -- <symbol> xxx / +xx% / x orders -- ...
        pnl = self.profit_and_loss()

        out = f"{colors.BOLD}Portfolio net: "
        out += print_item(
            currency=self._reference_currency,
            value=pnl["total_value"],
            profit_percentage=pnl["profit_percentage"],
            n_orders=pnl["n_orders"],
        )

        for _, symbol in self.symbols.items():
            out += f" --- {symbol.__str__(currency=self._reference_currency)}"

        return out


if __name__ == "__main__":

    pf = Portfolio(100000, results_parent_dir="./runs/backtest")

    for i in range(50):

        optimal_positions = {
            "BTC/EUR": 0,
            "ETH/EUR": 0,
        }

        if np.random.choice([False, True]):
            optimal_positions = {
                "BTC/EUR": np.random.uniform(high=10),
                "ETH/EUR": np.random.uniform(high=10),
            }

        tick = [
            {
                "trading_pair": "BTC/EUR",
                "open": 3902.52,
                "high": 3908.0,
                "low": 3902.25,
                "close": np.random.uniform(3500, 4000),
                "volume": 0.25119066,
                "timestamp": pd.Timestamp.now(),
            },
            {
                "trading_pair": "ETH/EUR",
                "open": 3902.52,
                "high": 3908.0,
                "low": 3902.25,
                "close": np.random.uniform(3500, 4000),
                "volume": 0.25119066,
                "timestamp": pd.Timestamp.now(),
            },
        ]

        orders = pf.update(tick, optimal_positions)

        prices = extract_prices(tick, "close")

        for order in orders:
            id = order["order_id"]
            price = prices[order["trading_pair"]]
            fee = 0  # price / 100
            ts = timestamp_to_string(pd.Timestamp.now())
            cost = price * order["amount"] + fee
            pf.settle_order(
                trading_pair=order["trading_pair"],
                order_id=id,
                price_settlement=price,
                timestamp_settlement=ts,
                fee=fee,
                fee_currency="EUR",
                order_value=cost,
            )

        print(pf)

    print(pf.get_optimal_positions().head(5))
    print(pf.get_orders().head(5))
    print(pf.get_settled_orders().head(5))
