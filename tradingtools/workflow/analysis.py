import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
from tqdm.notebook import tqdm_notebook as tqdm

from .labelling import create_signals


def plot_pairs(df, from_idx=0, to_idx=None, plot_standardized=True):

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
    n_plots = len(df["pair"].unique()) + plot_standardized
    fig, axes = plt.subplots(nrows=n_plots, figsize=(10, 3 * n_plots))

    for i, (pair, df_pair) in enumerate(df_plot.groupby("pair")):

        try:
            ax = axes[i + plot_standardized]
        except TypeError:
            ax = axes
        except Exception as e:
            raise e
            
        # All prices, standardized
        if plot_standardized and "standardized_price" in df_pair:
            df_pair["standardized_price"].plot(
                ax=axes[0], title="All - standardized", label=pair
            )

        # Plot pair
        df_pair["price"].plot(ax=ax, title=pair)

        # Plot signals
        if "signals" in df_pair:

            # Sell
            df_sell = df_pair["price"][df_pair["signals"] == "sell"]
            if len(df_sell) > 0:
                df_sell.plot(ax=ax, marker="v", linestyle="None", color="red")

            # Buy
            df_buy = df_pair["price"][df_pair["signals"] == "buy"]
            if len(df_buy):
                df_buy.plot(ax=ax, marker="^", linestyle="None", color="green")

        # Plot markers for gaps
        if "upcoming_gap" in df_pair:
            df_gaps = df_pair["price"][df_pair["upcoming_gap"]]
            if len(df_gaps) > 0:
                df_gaps.plot(ax=ax, marker="X", linestyle="None", color="orange")

        # Plot profit from backtest
        if "profit_percentage" in df_pair:
            line_color = "green" if df_pair["profit_percentage"][-1] > 0 else "red"
            df_pair["profit_percentage"].plot(ax=ax, color=line_color, secondary_y=True)

        # Align x-axis
        ax.set_xlim(dates["min_date"], dates["max_date"])

    # Align x-axis for All - standardized plot
    ax.set_xlim(dates["min_date"], dates["max_date"])

    ax.legend(loc="lower right")
    fig.tight_layout()

    return fig


def objective(params, df, history, ahead, cost_factor=0.002):

    """Objective for hyperparameter optimization using Ray tune

    Returns:
        float: profit over provided data window
    """

    # Create signals
    df["signals"] = create_signals(history, ahead, args=params)
    df["target_fraction"] = np.where(df["signals"] == "buy", 1, 0)

    profit_percentage = backtest(
        df, cost_factor=cost_factor, verbose=False, track_metrics=False
    )

    return profit_percentage


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

    def get_profit_percentage(self):
        return (
            (self.get_total_value() - self.starting_capital)
            / self.starting_capital
            * 100
        )

    def get_quantity(self):
        return self.quantity_asset


def backtest_pair(
    df_pair, starting_capital=1000, cost_factor=0.001, verbose=True, track_metrics=True
):

    portfolio = Portfolio(starting_capital, cost_factor)
    ticks_backtest = []

    ticks = df_pair.itertuples()

    if verbose:
        ticks = tqdm(ticks, total=len(df_pair))

    for tick in ticks:

        if tick.upcoming_gap and portfolio.get_quantity() > 0:
            portfolio.update(tick.price, "sell", 0)

        else:
            portfolio.update(tick.price, tick.signals, tick.target_fraction)

        if track_metrics:
            ticks_backtest.append(
                {
                    "datetime": tick.Index,
                    "value": portfolio.get_total_value(),
                    "profit_percentage": portfolio.get_profit_percentage(),
                    "quantity": portfolio.get_quantity(),
                }
            )

    if not track_metrics:
        return portfolio.get_profit_percentage()

    df_backtest = pd.DataFrame(ticks_backtest)
    df_backtest = df_backtest.set_index("datetime")
    return df_backtest


def backtest(
    df, starting_capital=1000, cost_factor=0.001, verbose=True, track_metrics=True
):

    results = []
    starting_capital_pair = starting_capital / df["pair"].nunique()

    for _, df_pair in df.groupby("pair"):

        backtest_result = backtest_pair(
            df_pair=df_pair,
            starting_capital=starting_capital_pair,
            cost_factor=cost_factor,
            verbose=verbose,
            track_metrics=track_metrics,
        )
        results.append(backtest_result)

    if not track_metrics:
        return np.mean(results)

    return pd.concat(results)


if __name__ == "__main__":

    p = Portfolio()
    df = pd.read_pickle("./notebooks/temp.pickle")

    backtest_pair(df, verbose=False)

    exit()

    p.update(100, "buy", 0.1)
    print(p.get_total_value())
    print(p.get_profit_percentage())
    p.update(100, "hold", 0)
    print(p.get_total_value())
    print(p.get_profit_percentage())
    p.update(100, "buy", 0.2)
    print(p.get_total_value())
    print(p.get_profit_percentage())
    p.update(1000, "buy", 0.2)
    print(p.get_total_value())
    print(p.get_profit_percentage())
    p.update(800, "hold", 0)
    print(p.get_total_value())
    print(p.get_profit_percentage())
    p.update(800, "hold", 0)
    print(p.get_total_value())
    print(p.get_profit_percentage())
    p.update(900, "sell", 0.1)
    print(p.get_total_value())
    print(p.get_profit_percentage())
    p.update(950, "sell", 0.1)
    print(p.get_total_value())
    print(p.get_profit_percentage())
    p.update(900, "sell", 0)
    print(p.get_total_value())
    print(p.get_profit_percentage())
    p.update(900, "sell", 0)
    print(p.get_total_value())
    print(p.get_profit_percentage())
