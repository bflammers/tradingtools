import pandas as pd
import numpy as np
import uuid


try:
    from .utils import (
        colors,
        extract_prices,
        warnings,
    )
except:
    from utils import (
        colors,
        extract_prices,
        warnings,
    )


class Portfolio:
    def __init__(self, start_capital) -> None:
        super().__init__()

        self._current_optimal_positions = []
        self._opt_positions_fixed_columns = ["positions_id", "timestamp"]
        self._opt_positions = pd.DataFrame(columns=self._opt_positions_fixed_columns)

        self._orders = pd.DataFrame(
            columns=[
                "order_id",
                "symbol",
                "status",
                "order_type",
                "volume",
                "timestamp",
                "timestamp_execution",
                "timestamp_settlement",
                "price_execution",
                "price_settlement",
                "fee",
                "slippage",
            ]
        )

        self._assets = dict()
        self._prices = dict()
        self._start_capital = start_capital
        self._unallocated_capital = start_capital

    def update(self, tick: list, optimal_positions: list = []) -> list:

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

        # Update latest prices of assets
        self._update_asset_prices(tick)

        # If optimal position is not equal to the current position we will update
        if optimal_positions != self._current_optimal_positions:

            timestamp = pd.Timestamp(tick[0]["timestamp"])

            # Update positions
            self._update_optimal_positions(optimal_positions, timestamp)

            # Set dict of deltas per symbol
            delta = self._calculate_delta()

            # Prepare order lists
            orders = self._construct_orders(delta, timestamp)

            return orders

        # Return empty orders list
        return []

    def _update_asset_prices(self, tick: list) -> None:

        # Extract closing prices
        self._prices = extract_prices(tick, "close")

        # Update asset prices
        for symbol, details in self._assets.items():

            # Determine new value based on prices and volume
            value_current = details["volume"] * self._prices[symbol]
            self._assets[symbol]["value_current"] = value_current

    def _update_optimal_positions(
        self, new_positions: dict, timestamp: pd.Timestamp
    ) -> None:

        """Internal method to update the intended positions. Note that this method
        only updates the "ideal" positions, and not the actual orders that will be
        created for the broker (these will be created through the _construct_orders method)
        """

        # Update fixed portfolio columns
        data_new_row = {
            "positions_id": uuid.uuid4().hex,
            "timestamp": timestamp,
        }

        # Update volume for each symbol, add new if not yet present
        for position in new_positions:
            data_new_row[position["symbol"]] = position["volume"]

        df_new = pd.DataFrame(data_new_row, index=[self._opt_positions.shape[0]])

        # Concat dataframes
        self._opt_positions = pd.concat([self._opt_positions, df_new], axis=0)

        # Fill NaNs with zeros (newly added symbols will have mostly NaNs)
        self._opt_positions = self._opt_positions.fillna(0)

        # Update current positions
        self._current_optimal_positions = new_positions

    def _calculate_delta(self) -> dict:

        """Internal method that calculates the delta with the previous positions
        (positions are the "optimal" positions, not actual assets) based on the
        portfolio dataframe

        Returns:
            dict: dict with difference in volume per asset with the previous position
        """

        # #### Delta with assets
        #
        # delta = dict()
        # current_pos = (
        #     self._opt_positions.drop(columns=self._opt_positions_fixed_columns)
        #     .iloc[-1]
        #     .to_dict()
        # )
        # for symbol in current_pos:
        #     if symbol not in self._assets:
        #         delta[symbol] = current_pos[symbol]
        #     else:
        #         delta[symbol] = current_pos[symbol] - self._assets[symbol]["volume"]
        # return delta

        if self._opt_positions.shape[0] <= 1:
            # Initial position is always buy
            delta = self._opt_positions.drop(columns=self._opt_positions_fixed_columns)
        else:
            delta = (
                self._opt_positions.drop(columns=self._opt_positions_fixed_columns)
                .tail(2)
                .diff(1)
                .tail(1)
            )

        delta = delta.squeeze(axis=0).to_dict()
        return delta

    def _construct_orders(self, delta: dict, timestamp: pd.Timestamp) -> list:

        """Internal method the contruct the orders from the deltas in position. The produces
        a list of buy / sell orders to be passed to the broker that actually submit the orders
        to the exchange. It includes the execution prices and timestamp in the order for
        future reference and slippage calculations.

        Returns:
            list: list of buy / sell orders to be passed to the broker
        """

        orders = []
        for symbol, volume_diff in delta.items():

            if volume_diff != 0:

                order_id = uuid.uuid4().hex
                order_type = "buy" if volume_diff > 0 else "sell"
                order_volume = abs(volume_diff)
                order_value = self._prices[symbol] * order_volume

                # Do not add order if not sufficient unallocated capital is available
                if order_type == "buy" and order_value > self._unallocated_capital:

                    warnings.warn(
                        "[Portfolio._construct_orders()]"
                        f"\n\t{colors.WARNING}Order not added (insufficient funds){colors.ENDC}"
                        f"\n\tOrder id: {order_id}"
                        f"\n\tSymbol: {symbol}"
                        f"\n\tPrice: {self._prices[symbol]}"
                        f"\n\tVolume: {order_volume}"
                        f"\n\tTotal order value: {order_value}"
                        f"\n\tUnallocated capital: {self._unallocated_capital}"
                    )

                    continue

                # Check for similar order already pending
                pending_orders = self._orders[self._orders["status"] == "pending"]
                matching_pending_orders = pending_orders[
                    (pending_orders["symbol"] == symbol)
                    & (pending_orders["symbol"] == order_type)
                ]

                if matching_pending_orders.shape[0] > 0:

                    warnings.warn(
                        "[Portfolio._construct_orders()]"
                        f"\n\t{colors.WARNING}Order not added ({matching_pending_orders.shape[0]} already pending){colors.ENDC}"
                        f"\n\tOrder id: {order_id}"
                        f"\n\tSymbol: {symbol}"
                        f"\n\tPrice: {self._prices[symbol]}"
                        f"\n\tVolume: {order_volume}"
                        f"\n\tTotal order value: {order_value}"
                        f"\n\tUnallocated capital: {self._unallocated_capital}"
                    )

                    continue

                # Append order to orders list
                orders.append(
                    {
                        "order_id": order_id,
                        "symbol": symbol,
                        "order_type": order_type,
                        "volume": order_volume,
                    }
                )

                # Construct new row as df
                df_new_row = pd.DataFrame(
                    {
                        "order_id": order_id,
                        "status": "pending",
                        "symbol": symbol,
                        "timestamp": timestamp,
                        "timestamp_execution": pd.Timestamp.now(),
                        "price_execution": self._prices[symbol],
                        "order_type": order_type,
                        "volume": order_volume,
                    },
                    index=[self._orders.shape[0]],
                )

                # Concat dataframes
                self._orders = pd.concat([self._orders, df_new_row], axis=0)

        return orders

    def add_settlement(
        self,
        order_id: str,
        price: float,
        fee: float,
        timestamp_settlement: pd.Timestamp,
    ) -> None:

        """Adds a settlement to an order and updates asset volumes and value at buy when the
        order has been filled. Uses the order_id to id the corresponding order in the orders df
        and updates with price (settlement) and timestamp (settlement)
        """

        # Add to orders dataframe
        order_idx = self._orders.index[self._orders["order_id"] == order_id]
        self._orders.loc[order_idx, "timestamp_settlement"] = timestamp_settlement
        self._orders.loc[order_idx, "price_settlement"] = price
        self._orders.loc[order_idx, "slippage"] = (
            price - self._orders.loc[order_idx, "price_execution"]
        )
        self._orders.loc[order_idx, "fee"] = fee

        # Extract requirements for updating assets
        volume_diff = self._orders.loc[order_idx, "volume"].values[0]
        value_diff = volume_diff * price
        symbol = self._orders.loc[order_idx, "symbol"].values[0]
        order_type = self._orders.loc[order_idx, "order_type"].values[0]

        # Update assets
        if order_type == "buy":

            # Add new asset if needed
            if symbol not in self._assets:
                self._assets[symbol] = {
                    "volume": 0,
                    "value_current": 0,
                    "total_value_at_buy": 0,
                    "total_value_at_sell": 0,
                }

            # Subtract from unallocated capital
            self._unallocated_capital -= value_diff

            # Add to symbol asset
            self._assets[symbol]["volume"] += volume_diff
            self._assets[symbol]["value_current"] += value_diff

            # Add to total bought value for this asset
            self._assets[symbol]["total_value_at_buy"] += value_diff

        elif order_type == "sell":

            # Add to unallocated capital
            self._unallocated_capital += value_diff

            # Subtract from symbol asset
            self._assets[symbol]["volume"] -= volume_diff
            self._assets[symbol]["value_current"] -= value_diff

            # Add to total sold value for this asset
            self._assets[symbol]["total_value_at_sell"] += value_diff

        else:

            # Order type should be either buy or sell
            raise Exception("order type not buy or sell")

        # Change order status to filled
        self._orders.loc[order_idx, "status"] = "filled"

    def get_optimal_positions(self) -> pd.DataFrame:
        return self._opt_positions

    def get_orders(self) -> pd.DataFrame:
        return self._orders

    def profit_and_loss(self) -> dict:

        """Method for retrieving the current profit and loss with the latest prices

        Returns:
            dict: overview of current pnl, overall and per asset
        """

        # n_orders = self.orders.groupby("symbol").size().to_dict()
        total_value = self._unallocated_capital

        pnl_assets = {}
        for symbol, asset_details in self._assets.items():
            total_value += asset_details["value_current"]

            asset_current_profit = (
                asset_details["value_current"]
                + self._assets[symbol]["total_value_at_sell"]
                - self._assets[symbol]["total_value_at_buy"]
            )

            pnl_assets[symbol] = asset_current_profit

        # Calculate profit percentage total
        profit_percentage_total = (
            (total_value - self._start_capital) / self._start_capital * 100
        )

        pnl = {
            "start_capital": self._start_capital,
            "unallocated": self._unallocated_capital,
            "total_value": total_value,
            "profit_percentage": profit_percentage_total,
            "assets": pnl_assets,
        }

        return pnl

    @staticmethod
    def _color_number_sign(x: float, decimals: int = 3, offset: float = 0) -> str:
        if (x - offset) > 0:
            return f"{colors.OKGREEN}+{x:.{decimals}f}{colors.ENDC}"
        else:
            return f"{colors.FAIL}{x:.{decimals}f}{colors.ENDC}"

    def _print_item(
        self, currency: float, value: float, profit_percentage: float = None
    ) -> str:

        value_colored = self._color_number_sign(value, decimals=2)
        out = f"{currency} {value_colored}"

        if profit_percentage is not None:
            profit_percentage_colored = self._color_number_sign(profit_percentage)
            out += f" / {profit_percentage_colored} %"

        return out

    def __str__(self) -> str:

        # Total: xxx / +xx% / x orders -- Unallocated: xxx -- <symbol> xxx / +xx% / x orders -- ...
        pnl = self.profit_and_loss()

        out = f"{colors.BOLD}Portfolio net: "
        out += self._print_item("USD", pnl["total_value"], pnl["profit_percentage"])

        for symbol, current_profit in pnl["assets"].items():
            out += f" --- {symbol}: "
            out += self._print_item("USD", current_profit)

        return out


if __name__ == "__main__":

    pf = Portfolio(100000)

    for i in range(50):

        optimal_positions = [
            {"symbol": "BTCUSD", "volume": 0},
            {"symbol": "ETHUSD", "volume": 0},
        ]

        if np.random.choice([False, True]):
            optimal_positions = [
                {"symbol": "BTCUSD", "volume": np.random.uniform(high=10)},
                {"symbol": "ETHUSD", "volume": np.random.uniform(high=10)},
            ]

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
            fee = price / 100
            ts = pd.Timestamp.now()
            pf.add_settlement(id, price, fee, ts)

        print(pf)
