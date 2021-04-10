
import numpy as np

from matplotlib import pyplot as plt

from tqdm.notebook import tqdm_notebook as tqdm


def plot_buy_sell(price, signal, gaps, from_idx=0, to_idx=None):

    buy, sell = signal == "buy", signal == "sell"
    
    if to_idx is None:
        to_idx = from_idx + 1000
    
    xx = np.arange(len(price))
    
    x1 = xx[from_idx:to_idx]
    y1 = price[from_idx:to_idx]

    x2 = x1[buy[from_idx:to_idx]]
    y2 = y1[buy[from_idx:to_idx]]

    x3 = x1[sell[from_idx:to_idx]]
    y3 = y1[sell[from_idx:to_idx]]

    x4 = x1[gaps[from_idx:to_idx]]
    y4 = y1[gaps[from_idx:to_idx]]

    plt.figure(figsize=(20, 8))
    plt.plot(x1, y1, color='orange')
    plt.plot(x2, y2, "^", color='green')
    plt.plot(x3, y3, "v", color='red')
    plt.plot(x4, y4, "X", color='blue')

def plot_price_profit(price, profit, from_idx=0, to_idx=None):
    
    if to_idx is None:
        to_idx = from_idx + 1000

    y1 = price[from_idx:to_idx]
    y2 = profit[from_idx:to_idx]

    fig, ax1 = plt.subplots(figsize=(20, 8))

    ax1.plot(y1)
    ax1.set_ylabel("price")

    ax2 = ax1.twinx()

    ax2.set_ylabel('profit') 
    ax2.plot(y2, color="orange")

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
