import inspect

import pandas as pd
import numpy as np

from .analysis import backtest

def judge_buy(
    history,
    ahead,
    b_irr=0.0015,
    b_n_irr=60,
    b_p_irr=0.1,
    b_nrr=0.0005,
    b_n_nrr=30,
    b_p_nrr=0.01,
    b_incr_nh=2,
    b_incr_na=2,
):

    current, history = history[:, -1], history[:, :-1]

    b_n_irr = np.clip(b_n_irr, 5, ahead.shape[1])
    b_n_nrr = np.clip(b_n_nrr, 5, ahead.shape[1])

    # In the money check
    irr_ahead = ahead[:, :b_n_irr] > (current * (1 + b_irr))[:, None]
    irr_check = irr_ahead.mean(axis=1) > b_p_irr

    # Negative rate check
    nrr_ahead = ahead[:, :b_n_nrr] < (current * (1 - b_nrr))[:, None]
    nrr_check = nrr_ahead.mean(axis=1) < b_p_nrr

    # Increasing check
    incr_history = (history[:, -b_incr_nh:] < current[:, None]).all(axis=1)
    incr_ahead = (current[:, None] < ahead[:, :b_incr_na]).all(axis=1)
    incr_check = incr_history & incr_ahead

    return irr_check & nrr_check & incr_check


def judge_sell(
    history,
    ahead,
    s_irr=0.0001,
    s_n_irr=30,
    s_p_irr=0.2,
    s_prr=0.0005,
    s_n_prr=60,
    s_p_prr=0.2,
    s_decr_nh=2,
    s_decr_na=2,
):

    current, history = history[:, -1], history[:, :-1]

    s_n_irr = np.clip(s_n_irr, 5, ahead.shape[1])
    s_n_prr = np.clip(s_n_prr, 5, ahead.shape[1])

    # In the money check
    irr_ahead = ahead[:, :s_n_irr] < (current * (1 - s_irr))[:, None]
    irr_check = irr_ahead.mean(axis=1) > s_p_irr

    # Positive rate check
    prr_ahead = ahead[:, :s_n_prr] > (current * (1 + s_prr))[:, None]
    prr_check = prr_ahead.mean(axis=1) < s_p_prr

    # Decreasing check
    decr_history = (history[:, -s_decr_nh:] > current[:, None]).all(axis=1)
    decr_ahead = (current[:, None] > ahead[:, :s_decr_na]).all(axis=1)
    decr_check = decr_history & decr_ahead

    return irr_check & prr_check & decr_check


def _add_kwargs(fn, kwargs=None):

    if kwargs is None:
        return fn

    # Kwargs contains more arguments (not just for this function)
    # so let's filter out the ones that apply to fn
    fn_args = list(inspect.signature(fn).parameters.keys())
    kwargs = {k: v for k, v in kwargs.items() if k in fn_args}
    
    def fn_wrapper(history, ahead):
        return fn(history, ahead, **kwargs)
        
    return fn_wrapper


def create_signals(history, ahead, args=None):

    judge_buy_wrapper = _add_kwargs(judge_buy, args)
    judge_sell_wrapper = _add_kwargs(judge_sell, args)

    buy = judge_buy_wrapper(history, ahead)
    sell = judge_sell_wrapper(history, ahead)

    targets = pd.Series(["hold"] * history.shape[0])
    targets[buy] = "buy"
    targets[sell] = "sell"

    return targets


def objective(params, price, history, ahead, cost_factor=0.0025):

    """Objective for hyperparameter optimization using Ray tune

    Returns:
        float: profit over provided data window
    """
    
    signals = create_signals(history, ahead, args=params)
    profit, _, _ = backtest(price, signals, cost_factor, verbose=False)
    
    return profit[-1]
