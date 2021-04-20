import numpy as np
import pandas as pd

from matplotlib import pyplot as plt

from tqdm.notebook import tqdm_notebook as tqdm


def plot_pairs(df, from_idx=0, to_idx=None):

    # Determine starting and stopping date
    df_subset = (
        df.reset_index(drop=False)
        .groupby("pair")
        .apply(lambda x: x.iloc[from_idx:to_idx])
        .reset_index(drop=True)
    )

    dates = (
        df_subset.groupby("pair")
        .agg(min_date=("datetime", np.min), max_date=("datetime", np.max))
        .median(numeric_only=False)
    )

    # Filter data between dates
    df_plot = df[(df.index > dates["min_date"]) & (df.index < dates["max_date"])]

    # Plotting
    n_plots = len(df["pair"].unique()) + 1
    fig, axes = plt.subplots(nrows=n_plots, figsize=(10, 3 * n_plots))

    for i, (pair, df_pair) in enumerate(df_plot.groupby("pair")):

        # All prices, standardized
        df_pair["standardized_price"].plot(
            ax=axes[0], title="All - standardized", label=pair
        )

        # Plot pair
        df_pair["price"].plot(ax=axes[i + 1], title=pair)

        # Plot signals
        if "signals" in df_pair:
            df_sell = df_pair["price"][df_pair["signals"] == "sell"]
            df_sell.plot(ax=axes[i + 1], marker="v", linestyle="None", color="red")
            df_buy = df_pair["price"][df_pair["signals"] == "buy"]
            df_buy.plot(ax=axes[i + 1], marker="^", linestyle="None", color="green")

        # Plot markers for gaps
        if "upcoming_gap" in df_pair:
            df_gaps = df_pair["price"][df_pair["upcoming_gap"]]
            df_gaps.plot(ax=axes[i + 1], marker="X", linestyle="None", color="orange")

        # Plot profit from backtest
        if "profit" in df_pair:
            line_color = "green" if df_pair["profit"][-1] > 0 else "red"
            df_pair["profit"].plot(ax=axes[i + 1], color=line_color, secondary_y=True)

        # Align x-axis
        axes[i + 1].set_xlim(dates["min_date"], dates["max_date"])

    # Align x-axis for All - standardized plot
    axes[0].set_xlim(dates["min_date"], dates["max_date"])

    axes[0].legend(loc="lower right")
    fig.tight_layout()


def backtest(price, signals, gaps, cost_factor=0.001, verbose=True):

    buy, sell = signals == "buy", signals == "sell"

    exposed = False
    buy_price = 0
    profit = 0
    n_orders = 0

    profit_over_time = np.zeros(len(price))
    exposed_over_time = np.zeros(len(price))

    if verbose:
        iter_price = enumerate(tqdm(price))
    else:
        iter_price = enumerate(price)

    for i, p in iter_price:

        if gaps[i]:

            # Make sure we are not exposed during time gaps
            if exposed:
                exposed = False
                profit += p - buy_price - cost_factor * (buy_price + p)
                n_orders += 1

        else:

            # Normal flow
            if buy[i] and not exposed:
                exposed = True
                buy_price = p
                n_orders += 1

            if sell[i] and exposed:
                exposed = False
                profit += p - buy_price - cost_factor * (buy_price + p)
                n_orders += 1

        profit_over_time[i] = profit
        exposed_over_time[i] = int(exposed)

    if verbose:
        print(f"Total profit: {profit}")
        print(f"Number of orders: {n_orders}")

    return profit_over_time, exposed_over_time, n_orders


def objective(params, price, history, ahead, gaps, cost_factor=0.002):

    """Objective for hyperparameter optimization using Ray tune

    Returns:
        float: profit over provided data window
    """

    signals = create_signals(history, ahead, args=params)
    profit, _, _ = backtest(price, signals, gaps, cost_factor, verbose=False)

    return profit[-1]


class Portfolio:
    def __init__(self, starting_capital=1000, cost_factor=0.001):

        # EVERYTHING EXPRESSED IN BASE CURRENCY
        self.starting_capital = starting_capital
        self.capital_unallocated = starting_capital
        self.capital_allocated = 0
        self.latest_price_asset = None

        self.current_target_fraction = 0
        self.cost_factor = cost_factor
        self.quantity_asset = 0

    def update(self, price, signal, target_fraction):

        assert price > 0
        assert signal in {"buy", "sell", "hold"}
        assert target_fraction >= 0 and target_fraction <= 1

        # Update price, capital and current fraction
        self.latest_price_asset = price
        self.capital_allocated = self.quantity_asset * price

        if (
            signal in ["buy", "sell"]
            and target_fraction != self.current_target_fraction
        ):

            # Calculate fraction of total value that is allocated
            total_value = self.get_total_value()
            actual_fraction_allocated = self.capital_allocated / total_value

            # Determine effective difference in fraction
            fraction_diff = target_fraction - actual_fraction_allocated

            # Determine value diffs
            order_value = fraction_diff * total_value
            abs_order_min_cost = abs(order_value) * (1 - self.cost_factor)

            # Buy
            if signal == "buy":
                assert order_value > 0, "Buy signal with non-positive order value"
                self.capital_unallocated -= abs(order_value)
                self.capital_allocated += abs_order_min_cost
                self.quantity_asset += abs_order_min_cost / price

            # Sell
            if signal == "sell":
                assert order_value < 0, "Sell signal with non-positive order value"
                self.capital_unallocated += abs_order_min_cost
                self.capital_allocated -= abs(order_value)
                self.quantity_asset -= abs(order_value) / price

            self.current_target_fraction = target_fraction

    def get_total_value(self):
        return self.capital_unallocated + self.capital_allocated

    def get_profit(self):
        return (
            (self.get_total_value() - self.starting_capital)
            / self.starting_capital
            * 100
        )

    def get_quantity(self):
        return self.quantity_asset


def backtest_pair(
    df, starting_capital=1000, cost_factor=0.001, verbose=True, track_metrics=True
):

    portfolio = Portfolio(starting_capital, cost_factor)
    ticks_backtest = []

    ticks = enumerate(df.itertuples())

    if verbose:
        ticks = tqdm(ticks, total=len(df))

    for i, tick in ticks:

        if tick.upcoming_gap and portfolio.get_quantity() > 0:
            portfolio.update(tick.price, "sell", 0)

        else:

            portfolio.update(tick.price, tick.signals, tick.target_fraction)

            if track_metrics:
                ticks_backtest.append(
                    {
                        "datetime": tick.Index,
                        "value": portfolio.get_total_value(),
                        "profit": portfolio.get_profit(),
                        "quantity": portfolio.get_quantity(),
                    }
                )

    if not track_metrics:
        return portfolio.get_profit()

    df_backtest = pd.DataFrame(ticks_backtest)
    df_backtest = df_backtest.set_index("datetime")
    return df_backtest


if __name__ == "__main__":

    p = Portfolio()
    df = pd.read_pickle("./notebooks/temp.pickle")

    backtest_pair(df, verbose=False)

    exit()

    p.update(100, "buy", 0.1)
    print(p.get_total_value())
    print(p.get_profit())
    p.update(100, "hold", 0)
    print(p.get_total_value())
    print(p.get_profit())
    p.update(100, "buy", 0.2)
    print(p.get_total_value())
    print(p.get_profit())
    p.update(1000, "buy", 0.2)
    print(p.get_total_value())
    print(p.get_profit())
    p.update(800, "hold", 0)
    print(p.get_total_value())
    print(p.get_profit())
    p.update(800, "hold", 0)
    print(p.get_total_value())
    print(p.get_profit())
    p.update(900, "sell", 0.1)
    print(p.get_total_value())
    print(p.get_profit())
    p.update(950, "sell", 0.1)
    print(p.get_total_value())
    print(p.get_profit())
    p.update(900, "sell", 0)
    print(p.get_total_value())
    print(p.get_profit())
    p.update(900, "sell", 0)
    print(p.get_total_value())
    print(p.get_profit())
