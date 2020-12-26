from numpy.core.fromnumeric import squeeze
import pandas as pd
import numpy as np
import uuid

from pandas.core import groupby


try:
    from .utils import colors, timestamp_to_string, string_to_timestamp
except:
    from utils import colors, timestamp_to_string, string_to_timestamp


class Portfolio:
    def __init__(self, start_capital) -> None:
        super().__init__()

        self._current_positions = []
        self._opt_positions_fixed_columns = ["positions_id", "timestamp"]
        self._opt_positions = pd.DataFrame(columns=self._opt_positions_fixed_columns)

        self._orders = pd.DataFrame(
            columns=[
                "order_id",
                "symbol",
                "timestamp_execution",
                "price_execution",
                "order_type",
                "volume",
                "timestamp_settlement",
                "price_settlement",
                "slippage",
            ]
        )

        self._assets = dict()
        self._start_capital = start_capital
        self._unallocated_capital = start_capital

    def update(self, prices: dict, optimal_positions: list = None) -> list:

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

        if (
            optimal_positions is not None
            and optimal_positions != self._current_positions
        ):

            # Update positions
            timestamp_execution = self._update_positions(optimal_positions, prices)

            # Set dict of deltas per symbol
            delta = self._calculate_delta()

            # Prepare order lists
            orders = self._construct_orders(delta, prices, timestamp_execution)

            return orders

        # Update assets using current prices
        self._update_asset_values(prices)

        # Return empty orders list
        return []

    def _update_positions(self, new_positions: dict, prices: dict) -> str:

        """Internal method to update the intended positions. Note that this method
        only updates the "ideal" positions, and not the actual orders that will be
        created for the broker (these will be created through the _construct_orders method)

        Returns:
            str: execution timestamp string
        """

        # Determine meta settings
        idx = self._opt_positions.shape[0]
        positions_id = uuid.uuid4().hex
        timestamp_execution = pd.Timestamp.now()

        # Update fixed portfolio columns
        self._opt_positions.loc[idx, "positions_id"] = positions_id
        self._opt_positions.loc[idx, "timestamp"] = timestamp_execution

        # Update volume for each symbol, add new if not yet present
        for position in new_positions:
            self._opt_positions.loc[idx, position["symbol"]] = position["volume"]

        # Fill NaNs with zeros (newly added symbols will have mostly NaNs)
        self._opt_positions = self._opt_positions.fillna(0)

        # Update current positions
        self._current_positions = new_positions

        return timestamp_to_string(timestamp_execution)

    def _update_asset_values(self, prices: dict) -> None:

        """Internal method to update the value of current assets based on new prices"""

        for symbol, details in self._assets.items():

            # Determine new value based on prices and volume
            value_current = self._assets[symbol]["volume"] * prices[symbol]
            self._assets[symbol]["value_current"] = value_current

            # Calculate the profit as a percentage of the starting value
            value_increase = value_current - self._assets[symbol]["value_at_buy"]
            self._assets[symbol]["profit_percentage"] = (
                value_increase / details["value_at_buy"] * 100
            )

    def _calculate_delta(self) -> dict:

        """Internal method that calculates the delta with the previous positions
        (positions are the "optimal" positions, not actual assets) based on the
        portfolio dataframe

        Returns:
            dict: dict with difference in volume per asset with the previous position
        """
        # TODO: check if this works correctly for assets that are added for the first time
        # Determine delta with previous positions
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

        return delta.squeeze(axis=0).to_dict()

    def _construct_orders(
        self, delta: dict, prices: dict, timestamp_execution: str
    ) -> list:

        """Internal method the contruct the orders from the deltas in position. The produces
        a list of buy / sell orders to be passed to the broker that actually submit the orders
        to the exchange. It includes the execution prices and timestamp in the order for
        future reference and slippage calculations.

        Returns:
            list: list of buy / sell orders to be passed to the broker
        """

        orders = []
        for symbol in delta:

            if delta[symbol] != 0:

                order_id = uuid.uuid4().hex
                order_type = "buy" if delta[symbol] > 0 else "sell"
                order_volume = abs(delta[symbol])
                order_value = prices[symbol] * order_volume

                # Concel order of  if order would surpass
                if order_type == "buy" and order_value > self._unallocated_capital:

                    Warning(
                        "[Portfolio._construct_orders()]"
                        f" - Order with id {order_id} not added to list - Symbol: {symbol}"
                        f" - Price: {prices[symbol]} - Volume: {order_volume}"
                        f" - Total order value: {order_value}"
                        f" - Unallocated capital: {self._unallocated_capital}"
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

                idx = self._orders.shape[0]

                self._orders.loc[idx, "order_id"] = order_id
                self._orders.loc[idx, "symbol"] = symbol
                self._orders.loc[idx, "timestamp_execution"] = string_to_timestamp(
                    timestamp_execution
                )
                self._orders.loc[idx, "price_execution"] = prices[symbol]
                self._orders.loc[idx, "order_type"] = order_type
                self._orders.loc[idx, "volume"] = order_volume

        return orders

    def add_settlement(self, order_id: str, price: float, timestamp: str) -> None:

        """Adds a settlement to an order and updates asset volumes and value at buy when the
        order has been filled. Uses the order_id to id the corresponding order in the orders df
        and updates with price (settlement) and timestamp (settlement)
        """

        # Add to orders dataframe
        idx = self._orders.index[self._orders["order_id"] == order_id]
        self._orders.loc[idx, "timestamp_settlement"] = timestamp
        self._orders.loc[idx, "price_settlement"] = price
        self._orders.loc[idx, "slippage"] = (
            price - self._orders.loc[idx, "price_execution"]
        )

        # Extract requirements for updating assets
        volume_diff = self._orders.loc[idx, "volume"].values[0]
        value_diff = volume_diff * price
        symbol = self._orders.loc[idx, "symbol"].values[0]
        order_type = self._orders.loc[idx, "order_type"].values[0]

        # Update assets
        if order_type == "buy":

            # Add new asset if needed
            if symbol not in self._assets:
                self._assets[symbol] = {
                    "volume": 0,
                    "value_at_buy": 0,
                    "value_current": 0,
                    "profit_percentage": 0,
                }

            # Subtract from unallocated capital
            self._unallocated_capital -= value_diff

            # Add to symbol asset
            self._assets[symbol]["volume"] += volume_diff
            self._assets[symbol]["value_current"] += value_diff
            self._assets[symbol]["value_at_buy"] += value_diff

        elif order_type == "sell":

            # Add to unallocated capital
            self._unallocated_capital += value_diff

            # Subtract from symbol asset
            volume_old = self._assets[symbol]["volume"]
            self._assets[symbol]["volume"] -= volume_diff
            self._assets[symbol]["value_current"] -= value_diff

            if self._assets[symbol]["volume"] == 0:
                # Reset value at buy
                self._assets[symbol]["value_at_buy"] = 0
            else:
                # TODO: nicer way of keeping value at buy?
                factor = self._assets[symbol]["volume"] / volume_old
                self._assets[symbol]["value_at_buy"] *= factor

        else:

            # Order type should be either buy or sell
            raise Exception("order type not buy or sell")

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

        for _, asset_details in self._assets.items():
            total_value += asset_details["value_current"]

        pnl = {
            "start_capital": self._start_capital,
            "unallocated": self._unallocated_capital,
            "total_value": total_value,
            "profit_percentage": (total_value - self._start_capital)
            / self._start_capital
            * 100,
        }

        return pnl

    @staticmethod
    def _color_number_sign(x: float, decimals: int = 3, offset: float = 0) -> str:
        if (x - offset) > 0:
            return f"{colors.OKGREEN}+{x:.{decimals}f}{colors.ENDC}"
        else:
            return f"{colors.FAIL}{x:.{decimals}f}{colors.ENDC}"

    def _print_item(
        self, currency: float, value: float, profit_percentage: float
    ) -> str:
        value_colored = self._color_number_sign(value, decimals=2)
        profit_percentage_colored = self._color_number_sign(profit_percentage)
        return f"{currency} {value_colored} / {profit_percentage_colored} %"

    def __str__(self) -> str:

        # Total: xxx / +xx% / x orders -- Unallocated: xxx -- <symbol> xxx / +xx% / x orders -- ...
        pnl = self.profit_and_loss()

        out = f"{colors.BOLD}Portfolio net: "
        out += self._print_item("USD", pnl["total_value"], pnl["profit_percentage"])

        for symbol, details in self._assets.items():
            out += f" --- {symbol}: "
            out += self._print_item(
                "USD", details["value_current"], details["profit_percentage"]
            )

        return out


if __name__ == "__main__":

    pf = Portfolio(1000)

    for i in range(5):

        optimal_positions = [
            {"symbol": "BTCUSD", "volume": np.random.uniform(high=10)},
            {"symbol": "ETHUSD", "volume": np.random.uniform(high=10)},
        ]

        if np.random.choice([False, True]):
            optimal_positions = [
                {"symbol": "BTCUSD", "volume": np.random.uniform(high=10)},
                {"symbol": "ETHUSD", "volume": np.random.uniform(high=10)},
            ]

        prices = {
            "BTCUSD": np.random.uniform(high=10),
            "ETHUSD": np.random.uniform(high=10),
        }

        orders = pf.update(prices, optimal_positions)

        for order in orders:
            id = order["order_id"]
            price = np.random.uniform(high=2000)
            ts = pd.Timestamp.now()
            pf.add_settlement(id, price, ts)

        print(pf)

    # print("\norders: \n", pf.orders.drop(columns=["order_id"]))
    # print("\norder: \n", orders)
    # print("\npositions: \n", pf.get_history())
    # print("\nassets: \n", pf.assets)

    # pf.profit_and_loss()

    # start_positions = [
    #     {
    #         "symbol": "BTCUSD",
    #         "timestamp_execution": pd.Timestamp("2017-01-01 12:00:00"),
    #         "volume": 10,
    #         "price_execution": 100,
    #         "timestamp_settlement": pd.Timestamp("2017-01-01 12:00:02"),
    #         "price_settlement": 101,
    #     },
    #     {
    #         "symbol": "ETHUSD",
    #         "timestamp_execution": pd.Timestamp("2017-01-03 12:00:00"),
    #         "volume": 10,
    #         "price_execution": 120,
    #         "timestamp_settlement": pd.Timestamp("2017-01-03 12:00:02"),
    #         "price_settlement": 121,
    #     },
    #     {
    #         "symbol": "ETHUSD",
    #         "timestamp_execution": pd.Timestamp("2017-01-10 12:00:00"),
    #         "volume": -10,
    #         "price_execution": 120,
    #         "timestamp_settlement": pd.Timestamp("2017-01-10 12:00:02"),
    #         "price_settlement": 101,
    #     },
    #     {
    #         "symbol": "BTCUSD",
    #         "timestamp_execution": pd.Timestamp("2017-01-21 12:00:00"),
    #         "volume": -5,
    #         "price_execution": 90,
    #         "timestamp_settlement": pd.Timestamp("2017-01-21 12:00:02"),
    #         "price_settlement": 87,
    #     },
    # ]

    # df_start = pd.DataFrame(start_positions)
    # pf = Portfolio(df_start)
    # df_pos = pf.get_positions()
    # print(df_pos)
    # print(pf)
