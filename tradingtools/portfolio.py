import pandas as pd
import numpy as np
import typing

from uuid import uuid4
from decimal import Decimal

try:
    from .broker import Order, Settlement
    from .utils import (
        colors,
        extract_prices,
        warnings,
        timestamp_to_string,
        print_item,
    )
except:
    from broker import Order, Settlement
    from utils import (
        colors,
        extract_prices,
        warnings,
        timestamp_to_string,
        print_item,
    )


class Symbol:
    _open_orders: typing.Dict[str, Order]

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
        self._min_order_amount = Decimal("0.001")

    def sync_state(
        self,
        tick_timestamp: str = None,
        price: Decimal = None,
        current_amount: Decimal = None,
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

            # Convert to decimal
            current_amount = Decimal(current_amount)

            if current_amount != self._current_amount:

                # Update total value at buy/sell with Delta
                diff_amount = current_amount - self._current_amount
                self._total_value_at_buy += self._latest_price * diff_amount

                # Update
                self._current_amount = current_amount

    def update_optimal_position(self, optimal_amount: Decimal) -> Order:

        if abs(optimal_amount - self.optimal_amount) >= self._min_order_amount:

            # Construct order - NOT DELTA WITH CURRENT AMOUNT - LEADS TO MINIMAL AMOUNT ISSUES AT EXCHANGE
            delta = optimal_amount - (self._current_amount + self._pending_delta_amount)
            side = "buy" if delta > 0 else "sell"
            amount = abs(delta)

            # Create order
            order = self._create_order(amount, side)

            # Update state
            self.optimal_amount = optimal_amount
            self._open_orders[order.order_id] = order
            self._n_orders += 1
            self._pending_delta_amount += delta

            return order

        return None

    def settle_and_retrieve_order(self, settlement: Settlement) -> Order:

        # Retrieve open order and delete from open orders
        order = self._open_orders.pop(settlement.order_id)

        # Settle order
        order.settle(settlement=settlement)

        # Update symbol state
        if order.side == "buy":
            self._current_amount += Decimal(order.amount)
            self._pending_delta_amount -= Decimal(order.amount)
            self._total_value_at_buy += Decimal(order.value_settlement)
        elif order.side == "sell":
            self._current_amount -= Decimal(order.amount)
            self._pending_delta_amount += Decimal(order.amount)
            self._total_value_at_sell += Decimal(order.value_settlement)
        else:
            raise Exception(
                f"[Portfolio.Symbol] side {order.side} not known should be buy or sell"
            )

        return order

    def _create_order(
        self, amount: Decimal, side: str, quote_currency: str = "EUR"
    ) -> Order:

        if self._latest_price is None:
            raise Exception("[Symbol._create_order] price not set through sync_state")

        order = Order(
            order_id=uuid4().hex,
            trading_pair=f"{self.symbol_name}/{quote_currency}",
            status="pending",
            side=side,
            amount=Decimal(amount),
            timestamp_tick=self._latest_tick_timestamp,
            price_execution=Decimal(self._latest_price),
            cost_execution=Decimal(amount) * Decimal(self._latest_price),
            timestamp_execution=timestamp_to_string(pd.Timestamp.now()),
        )

        return order

    def get_current_value(self) -> Decimal:

        # Calculate current value and profit so far
        current_value = self._current_amount * self._latest_price

        return current_value

    def profit_and_loss(self) -> dict:

        current_value = self.get_current_value()
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
    symbols: typing.Dict[str, Symbol]

    def __init__(
        self,
        starting_capital: Decimal = None,
        reference_currency: str = "EUR",
        verbose: bool = True,
    ) -> None:
        super().__init__()

        self._verbose = verbose
        self._reference_currency = reference_currency
        self.symbols = dict()

        # Initialize state objects
        self._current_optimal_positions = {}
        self._reserved_capital = Decimal(0)

        self._starting_capital = starting_capital
        self._unallocated_capital = starting_capital

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

        if self._starting_capital is None:
            raise Exception("No starting capital --> first call initialize")

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
                    if order.side == "buy":
                        self._reserved_capital += order.cost_execution
                        self._unallocated_capital -= order.cost_execution

        # Update optimal positions
        self._current_optimal_positions = optimal_positions

        return orders

    def initialize(self, symbol_amounts: dict, tick: list) -> None:

        self.sync_prices(tick)  # Should be called before sync_amounts
        self.sync_amounts(symbol_amounts)

        starting_capital = 0

        for symbol in self.symbols.values():

            starting_capital += symbol.get_current_value()

        self._unallocated_capital = symbol_amounts[self._reference_currency]
        self._starting_capital = starting_capital + self._unallocated_capital

    def sync_prices(self, tick: list) -> None:

        for t in tick:

            trading_pair = t["trading_pair"]
            base, quote = trading_pair.split("/")

            if base not in self.symbols:
                self.symbols[base] = Symbol(base)

            price, timestamp = self._extract_price_from_tick(tick, trading_pair)
            symbol = self.symbols[base]

            symbol.sync_state(tick_timestamp=timestamp, price=price)

    def sync_amounts(self, symbol_amounts: dict) -> None:

        for symbol_name, amount in symbol_amounts.items():

            if symbol_name != self._reference_currency:

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

    def settle_order(self, settlement: Settlement) -> dict:

        """Adds a settlement to an order and updates asset volumes and value at buy when the
        order has been filled. Uses the order_id to id the corresponding order in the orders df
        and updates with price (settlement) and timestamp (settlement)
        """

        # Split trading pair
        base, quote = settlement.trading_pair.split("/")

        # Get symbol
        symbol = self.symbols[base]

        # Settle order
        settled_order = symbol.settle_and_retrieve_order(settlement=settlement)

        # Update unallocated and reserved capital
        if settled_order.side == "buy":
            # Free reserved capital only if side was buy
            self._reserved_capital -= settled_order.cost_execution
        elif settled_order.side == "sell":
            # Add to unallocated capital
            self._unallocated_capital += settled_order.value_settlement
        else:
            raise Exception("[Portfolio.settle_order] side order not buy or sell")

        return settled_order

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
            (current_value - self._starting_capital) / self._starting_capital * 100
        )

        pnl = {
            "start_capital": self._starting_capital,
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

    s = Symbol("BTC")
    s.sync_state(tick_timestamp="2020...", price=Decimal("1"))
    print(s)

    pf = Portfolio(starting_capital=Decimal(10000))

    for i in range(50):

        optimal_positions = {
            "BTC/EUR": Decimal(0),
            "ETH/EUR": Decimal(0),
        }

        if np.random.choice([False, True]):
            optimal_positions = {
                "BTC/EUR": Decimal(np.random.uniform(high=10)),
                "ETH/EUR": Decimal(np.random.uniform(high=10)),
            }

        tick = [
            {
                "trading_pair": "BTC/EUR",
                "open": Decimal(3902.52),
                "high": Decimal(3908.0),
                "low": Decimal(3902.25),
                "close": Decimal(np.random.uniform(3500, 4000)),
                "volume": Decimal(0.25119066),
                "timestamp": pd.Timestamp.now(),
            },
            {
                "trading_pair": "ETH/EUR",
                "open": Decimal(3902.52),
                "high": Decimal(3908.0),
                "low": Decimal(3902.25),
                "close": Decimal(np.random.uniform(3500, 4000)),
                "volume": Decimal(0.25119066),
                "timestamp": pd.Timestamp.now(),
            },
        ]

        orders = pf.update(tick, optimal_positions)

        prices = extract_prices(tick, "close")

        for order in orders:
            settlement = Settlement(
                order_id=order.order_id,
                trading_pair=order.trading_pair,
                status="completed",
                price=Decimal(prices[order.trading_pair]),
                fee=Decimal(0),
                timestamp=timestamp_to_string(pd.Timestamp.now()),
                cost=Decimal(prices[order.trading_pair]) * order.amount,
                value=Decimal(prices[order.trading_pair]) * order.amount
            )
            pf.settle_order(settlement)

        print(pf)
