
import numpy as np

from matplotlib import pyplot as plt

from tqdm.notebook import tqdm_notebook as tqdm


def plot_buy_sell(price, signal, from_idx=0, to_idx=None):

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

    plt.figure(figsize=(20, 8))
    plt.plot(x1, y1, color='orange')
    plt.plot(x2, y2, "^", color='green')
    plt.plot(x3, y3, "v", color='red')

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

def backtest(price, signals, cost_factor=0.001, starting_capital=None, verbose=True):
    
    buy, sell = signals == "buy", signals == "sell"
    
    if starting_capital is None:
        try:
            starting_capital = price[buy][0]
        except IndexError:
            starting_capital = 0
    
    exposed = False
    unallocated = starting_capital
    allocated = 0
    
    tv_at_buy = 0
    tv_at_sell = 0
    
    tv_costs = 0
    
    buy_price = 0
    profit = 0
    
    capital = np.zeros(len(price))
    amount = np.zeros(len(price))
    
    capital[0] = starting_capital

    n_orders = 0

    if verbose: 
        iter_price = enumerate(tqdm(price))
    else:
        iter_price = enumerate(price)

    for i, p in iter_price:

        if buy[i] and not exposed:
            exposed = True
            
            buy_price = p
            
            tv_at_buy += p
            
            unallocated -= p

            tv_costs += cost_factor * p

            n_orders += 1

        if sell[i] and exposed:
            exposed = False
            
            profit += p - buy_price
            
            tv_at_sell += p
            
            unallocated += p

            tv_costs += cost_factor * p

            n_orders += 1

        allocated = exposed * p

        capital[i] = allocated + unallocated - tv_costs
        amount[i] = int(exposed)
    
    if verbose:
        print(f"Total profit: {unallocated + allocated - starting_capital - tv_costs}")
        print(f"Total profit: {profit - tv_costs}")
        print(f"Total profit: {tv_at_sell - tv_at_buy - tv_costs + exposed * p}")
        print(f"Number of orders: {n_orders}")
    
    
    return capital - starting_capital, amount