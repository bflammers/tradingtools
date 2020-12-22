
import pandas as pd
import numpy as np

try:
    from .utils import colors
except:
    from utils import colors


class Portfolio:

    def __init__(self) -> None:
        super().__init__()


class OldPortfolio:
    def __init__(self, start_capital=0, start_orders=None):

        self.order_columns = [
            "symbol",
            "timestamp_execution",
            "price_execution",
            "order_type",
            "volume",
            "timestamp_settlement",
            "price_settlement",
            "slippage",
            "unallocated_capital",
            "total_capital"
        ]
        self.df_orders = pd.DataFrame(columns=self.order_columns)
        self.unallocated_capital = start_capital
        self.total_capital = None

        if start_orders is not None:
            assert set(start_orders.columns) == set(self.order_columns)
            self.df_orders = start_orders[self.order_columns]

    def get_orders(self):

        # Make copy of position dataframe
        df_orders = self.df_orders.copy()

        # Add columns
        df_orders["total_diff"] = df_orders["volume"] * df_orders["price_settlement"]
        df_orders["price_slippage"] = (
            df_orders["price_settlement"] - df_orders["price_execution"]
        )
        df_orders["total_slippage"] = df_orders["volume"] * df_orders["price_slippage"]

        return df_orders

    def add_order(
        self,
        symbol,
        timestamp_execution,
        order_type,
        volume,
        price_execution,
        timestamp_settlement,
        price_settlement,
    ):

        

        self.df_orders = self.df_orders.append(
            pd.Series(
                {
                    "symbol": symbol,
                    "timestamp_execution": timestamp_execution,
                    "price_execution": price_execution, 
                    "order_type": order_type,
                    "volume": volume,
                    "timestamp_settlement": timestamp_settlement,
                    "price_settlement": price_settlement,
                    "slippage": None,
                    "unallocated_capital": 
                    "total_capital"
                }
            ),
            ignore_index=True,
        )

        df_pos = self.get_positions()

        if not all(df_pos["volume_sum"] >= 0):
            raise Exception("Cannot take negative position")

    def add_order_settlement(self):
        raise NotImplementedError

    def get_positions(self):
        df_orders = self.get_orders()

        pos_columns = ["symbol", "volume", "total_diff", "total_slippage"]
        df_orders_grouped = df_orders[pos_columns].groupby("symbol")
        df_pos = df_orders_grouped.agg({
            'volume': ['sum', 'count'],
            'total_diff': 'sum',
            'total_slippage': 'sum'
        })
        df_pos.columns = df_pos.columns.map('_'.join)
        df_pos = df_pos.rename(columns={'volume_count': 'n_orders'})
        df_pos = df_pos.reset_index()
        return df_pos

    def _color_number_sign(self, x):
        if x > 0:
            return f"{colors.OKGREEN}+{x}{colors.ENDC}"
        else:
            return f"{colors.FAIL}{x}{colors.ENDC}"

    def __str__(self) -> str:

        df_pos = self.get_positions()
        diff_total = df_pos['total_diff_sum'].sum()
        n_orders_total = df_pos["n_orders"].sum()

        out = f'{colors.BOLD}Portfolio net: {self._color_number_sign(diff_total)}'
        out += f', {n_orders_total} orders '

        for _, row in df_pos.iterrows():
            out += f'--- {row["symbol"]}: {self._color_number_sign(row["total_diff_sum"])}'
            out += f', {row["n_orders"]} orders '

        return out


if __name__ == "__main__":

    start_positions = [
        {
            "symbol": "BTCUSD",
            "timestamp_execution": pd.Timestamp("2017-01-01 12:00:00"),
            "volume": 10,
            "price_execution": 100,
            "timestamp_settlement": pd.Timestamp("2017-01-01 12:00:02"),
            "price_settlement": 101,
        },
        {
            "symbol": "ETHUSD",
            "timestamp_execution": pd.Timestamp("2017-01-03 12:00:00"),
            "volume": 10,
            "price_execution": 120,
            "timestamp_settlement": pd.Timestamp("2017-01-03 12:00:02"),
            "price_settlement": 121,
        },
        {
            "symbol": "ETHUSD",
            "timestamp_execution": pd.Timestamp("2017-01-10 12:00:00"),
            "volume": -10,
            "price_execution": 120,
            "timestamp_settlement": pd.Timestamp("2017-01-10 12:00:02"),
            "price_settlement": 101,
        },
        {
            "symbol": "BTCUSD",
            "timestamp_execution": pd.Timestamp("2017-01-21 12:00:00"),
            "volume": -5,
            "price_execution": 90,
            "timestamp_settlement": pd.Timestamp("2017-01-21 12:00:02"),
            "price_settlement": 87,
        },
    ]

    df_start = pd.DataFrame(start_positions)
    pf = Portfolio(df_start)
    df_pos = pf.get_positions()
    print(df_pos)
    print(pf)
