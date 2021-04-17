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
        
        # Align x-axis
        axes[i + 1].set_xlim(dates["min_date"], dates["max_date"])
    
    # Align x-axis for All - standardized plot
    axes[0].set_xlim(dates["min_date"], dates["max_date"])

    axes[0].legend(loc="lower right")
    fig.tight_layout()


def backtest(df, signals, cost_factor=0.001, verbose=True):

    buy, sell = signals == "buy", signals == "sell"

    exposed = False
    buy_price = 0
    profit = 0
    n_orders = 0

    profit_over_time = np.zerosa
    (len(price))
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
