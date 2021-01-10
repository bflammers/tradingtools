import pandas as pd
import numpy as np
import uuid

from uuid import uuid4
from decimal import Decimal


try:
    from .utils import (
        warnings,
        timestamp_to_string,
        print_item,
    )
except:
    from utils import (
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

    def _create_order(
        self, amount: Decimal, side: str, quote_currency: str = "EUR"
    ) -> dict:

        order = {
            "order_id": uuid.uuid4().hex,
            "trading_pair": f"{self.symbol_name}/{quote_currency}",
            "side": side,
            "amount": Decimal(amount),
            "timestamp_tick": self._latest_tick_timestamp,
            "price_execution": Decimal(self._latest_price),
            "cost_execution": Decimal(amount) * Decimal(self._latest_price),
            "timestamp_execution": timestamp_to_string(pd.Timestamp.now()),
        }

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
