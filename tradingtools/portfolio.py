import pandas as pd
import numpy as np
import uuid
import csv

from uuid import uuid4
from decimal import Decimal
from pathlib import Path

from pandas.core.frame import DataFrame


try:
    from .utils import (
        colors,
        extract_prices,
        warnings,
        timestamp_to_string,
        print_item,
    )
except:
    from utils import (
        colors,
        extract_prices,
        warnings,
        timestamp_to_string,
        print_item,
    )


class Symbol:
    def __init__(self, symbol_name: str) -> None:
        super().__init__()
        self.symbol_name = symbol_name
        self.optimal_amount = Decimal(0)
        self._current_amount = Decimal(0)
        self._pending_delta_amount = Decimal(0)
        self._open_orders = {}
        self._latest_price = Decimal(0)
        self._latest_tick_timestamp = None
        self._n_orders = 0
        self._total_value_at_buy = Decimal(0)
        self._total_value_at_sell = Decimal(0)

    def sync_state(
        self, tick_timestamp: str, price: Decimal = None, current_amount: Decimal = None
    ) -> None:

        if tick_timestamp is not None:
            self._latest_tick_timestamp = tick_timestamp

        if price is not None:

            if tick_timestamp is None:
                warnings.warn(
                    "[Portfolio.Symbol] tick_timestamp should be provided with latest price"
                )

            self._latest_price = Decimal(price)

        if current_amount is not None:

            if tick_timestamp is not None:
                warnings.warn(
                    "[Portfolio.Symbol] tick_timestamp should only be provided with latest price, not with current_amount"
                )

            self._current_amount = Decimal(current_amount)

    def update_optimal_position(self, optimal_amount: Decimal) -> dict:

        if optimal_amount != self.optimal_amount:

            # Construct order
            delta = Decimal(optimal_amount) - (
                self._current_amount + self._pending_delta_amount
            )
            side = "buy" if delta > 0 else "sell"
            amount = abs(delta)
            order = self._create_order(amount, side)

            # Update state
            self.optimal_amount = optimal_amount
            self._open_orders[order["order_id"]] = order
            self._n_orders += 1
            self._pending_delta_amount += delta

            return order

        return None

    def add_settlement(self, order_id: str, order_value: Decimal):

        # Retrieve open order and delete from open orders
        order = self._open_orders.pop(order_id)

        # Update symbol state
        if order["side"] == "buy":
            self._current_amount += Decimal(order["amount"])
            self._pending_delta_amount -= Decimal(order["amount"])
            self._total_value_at_buy += Decimal(order_value)
        elif order["side"] == "sell":
            self._current_amount -= Decimal(order["amount"])
            self._pending_delta_amount += Decimal(order["amount"])
            self._total_value_at_sell += Decimal(order_value)
        else:
            raise Exception(
                f"[Portfolio.Symbol] side {order['side']} not knownm should be buy or sell"
            )

        return order

    def _create_order(self, amount: Decimal, side: str) -> dict:

        order = {
            "order_id": uuid.uuid4().hex,
            "symbol": self.symbol_name,
            "side": side,
            "amount": Decimal(amount),
            "timestamp_tick": self._latest_tick_timestamp,
            "price_execution": Decimal(self._latest_price),
            "cost_execution": Decimal(amount) * Decimal(self._latest_price),
            "timestamp_execution": timestamp_to_string(pd.Timestamp.now()),
        }

        return order

    def profit_and_loss(self) -> dict:

        # Calculate current value and profit so far
        current_value = self._current_amount * self._latest_price
        current_profit = (
            current_value + self._total_value_at_sell - self._total_value_at_buy
        )

        # Collect in dict
        pnl = {
            "amount": self._current_amount,
            "value": current_value,
            "profit": current_profit,
            "n_orders": self._n_orders,
            "n_open_orders": len(self._open_orders),
            "timestamp_valuation": self._latest_tick_timestamp,
        }

        return pnl

    def __str__(self, currency: str = "EUR") -> str:
        pnl = self.profit_and_loss()

        out = f"{self.symbol_name}: "
        out += print_item(
            currency=currency,
            value=pnl["value"],
            profit=pnl["profit"],
            n_orders=pnl["n_orders"],
        )

        return out


class Portfolio:
    def __init__(
        self,
        start_capital: Decimal,
        results_parent_dir: str = "./runs",
        base_currency: str = "EUR",
        verbose: bool = True
    ) -> None:
        super().__init__()

        self._verbose = verbose
        self._base_currency = base_currency
        self.symbols = {}

        # Create directory for results
        now = pd.Timestamp.now().strftime('%Y%m%d_%H%M')
        ts_uuid = f"{now}_{uuid4().hex}"
        self._results_dir = (Path(results_parent_dir) / ts_uuid).absolute()
        self._results_dir.mkdir(parents=True, exist_ok=True)

        if self._verbose:
            print(f"[Portfolio] results directory created: {self._results_dir}")

        # Create csv file and path for optimal_positions
        self._opt_positions_path = self._results_dir / f"{now}_optimal_positions.csv"
        self._opt_positions_columns = ["positions_id", "timestamp", "symbol", "amount"]

        with open(self._opt_positions_path, "w") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(self._opt_positions_columns)

        # Create csv file and path for orders
        self._orders_path = self._results_dir / f"{now}_orders.csv"
        self._orders_columns = [
            "order_id",
            "symbol",
            "side",
            "amount",
            "timestamp_tick",
            "price_execution",
            "cost_execution",
            "timestamp_execution",
        ]

        with open(self._orders_path, "w") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(self._orders_columns)

        # Create csv file and path for orders
        self._settlements_path = self._results_dir / f"{now}_settlements.csv"
        self._settlements_columns = [
            "order_id",
            "symbol",
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
            "fee_base_currency",
            "slippage",
            "order_value",
        ]

        with open(self._settlements_path, "w") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(self._settlements_columns)

        # Initialize state objects
        self._current_optimal_positions = {}
        self._start_capital = start_capital
        self._unallocated_capital = start_capital
        self._reserved_capital = Decimal(0)

    def _append_to_csv(self, target_csv: str, new_values: dict) -> None:

        if target_csv == "opt_positions":
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
                warnings.warn(f"[Portfolio._append_to_csv] key-value pair for {column} not in new values for {target_csv}")

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

        # Construct orders list to append orders to
        orders = []

        # Update positions
        for symbol_name, amount in optimal_positions.items():

            # Add symbol if needed
            if symbol_name not in self.symbols:
                self.symbols[symbol_name] = Symbol(symbol_name)
                self._current_optimal_positions[symbol_name] = 0

            # Load symbol
            symbol = self.symbols[symbol_name]

            # Update price and tick timestamp
            price, timestamp = self._extract_price_from_tick(tick, symbol_name)
            self._update_symbol_price(symbol, price, timestamp)

            if amount != symbol.optimal_amount:

                # Update symbol position, generate order if needed
                order = symbol.update_optimal_position(amount)

                if order is not None:

                    # Append orders if not None
                    orders.append(order)

                    # Reserve capital if buy
                    if order["side"] == "buy":
                        self._reserved_capital += order["cost_execution"]

        # Write optimal postions and orders to file
        self._write_optimal_positions(optimal_positions)
        self._write_orders(orders)

        return orders

    @staticmethod
    def _update_symbol_price(symbol: Symbol, price: Decimal, timestamp: str) -> None:
        symbol.sync_state(tick_timestamp=timestamp, price=price)

    @staticmethod
    def _extract_price_from_tick(
        tick: list, symbol_name: str, price_type: str = "close"
    ) -> tuple:
        for t in tick:
            if t["symbol"] == symbol_name:
                return t[price_type], t["timestamp"]

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
        symbol_name: str,
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

        # Get symbol
        symbol = self.symbols[symbol_name]

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
        if fee_currency != self._base_currency:
            fee_base_currency = Decimal(price_settlement) * Decimal(fee)
            settled_order["fee_base_currency"] = fee_base_currency
        else:
            settled_order["fee_base_currency"] = fee

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
            raise Exception('[Portfolio.settle_order] side order not buy or sell')

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
            currency=self._base_currency,
            value=pnl["total_value"],
            profit_percentage=pnl["profit_percentage"],
            n_orders=pnl["n_orders"],
        )

        for _, symbol in self.symbols.items():
            out += f" --- {symbol.__str__(currency=self._base_currency)}"

        return out


if __name__ == "__main__":

    pf = Portfolio(100000)

    for i in range(50):

        optimal_positions = {
            "BTCUSD": 0,
            "ETHUSD": 0,
        }

        if np.random.choice([False, True]):
            optimal_positions = {
                "BTCUSD": np.random.uniform(high=10),
                "ETHUSD": np.random.uniform(high=10),
            }

        tick = [
            {
                "symbol": "BTCUSD",
                "open": 3902.52,
                "high": 3908.0,
                "low": 3902.25,
                "close": np.random.uniform(3500, 4000),
                "volume": 0.25119066,
                "timestamp": pd.Timestamp.now(),
            },
            {
                "symbol": "ETHUSD",
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
            price = prices[order["symbol"]]
            fee = 0  # price / 100
            ts = timestamp_to_string(pd.Timestamp.now())
            cost = price * order["amount"] + fee
            pf.settle_order(
                symbol_name=order["symbol"],
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
