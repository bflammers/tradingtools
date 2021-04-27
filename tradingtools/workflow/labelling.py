import inspect

import pandas as pd
import numpy as np


def judge_buy(
    history,
    ahead,
    b_rate=0.003,
    b_n_rate=190,
    b_p_rate=0.3,
    b_neg_rate_allowed=0.001,
    b_n_incr_before=3,
    b_n_incr_after=7,
):

    # Divide history into current price and actual history
    current, history = history[:, -1], history[:, :-1]

    # In the money check
    rate_ahead = ahead[:, :b_n_rate] > (current * (1 + b_rate))[:, None]
    rate_check = rate_ahead.mean(axis=1) > b_p_rate

    # Negative rate check
    neg_rate_ahead = ahead[:, :b_n_rate] < (current * (1 - b_neg_rate_allowed))[:, None]
    neg_rate_check = ~np.any(neg_rate_ahead, axis=1)

    # Increasing check
    incr_history = (history[:, -b_n_incr_before:] < current[:, None]).all(axis=1)
    incr_ahead = (current[:, None] < ahead[:, :b_n_incr_after]).all(axis=1)
    incr_check = incr_history & incr_ahead

    return rate_check & neg_rate_check & incr_check


def judge_sell(
    history,
    ahead,
    s_rate=0.003,
    s_n_rate=190,
    s_p_rate=0.3,
    s_pos_rate_allowed=0.001,
    s_n_decr_before=3,
    s_n_decr_after=7,
):

    # Divide history into current price and actual history
    current, history = history[:, -1], history[:, :-1]

    # In the money check
    rate_ahead = ahead[:, :s_n_rate] < (current * (1 - s_rate))[:, None]
    rate_check = rate_ahead.mean(axis=1) > s_p_rate

    # Positive rate check
    pos_rate_ahead = ahead[:, :s_n_rate] > (current * (1 + s_pos_rate_allowed))[:, None]
    pos_rate_check = ~np.any(pos_rate_ahead, axis=1)

    # Decreasing check
    decr_before = (history[:, -s_n_decr_before:] > current[:, None]).all(axis=1)
    decr_after = (current[:, None] > ahead[:, :s_n_decr_after]).all(axis=1)
    decr_check = decr_before & decr_after

    return rate_check & pos_rate_check & decr_check


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

    targets = np.full(len(history), "hold")
    targets[buy] = "buy"
    targets[sell] = "sell"

    return targets
