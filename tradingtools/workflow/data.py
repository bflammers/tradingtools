import re
import multiprocessing

import pandas as pd
import numpy as np

from pathlib import Path

from tqdm.notebook import tqdm_notebook as tqdm


def _load_ticks(dir_name, pairs):

    binance_path = Path.cwd().parent / "data/collected/binance"
    data_path = binance_path / dir_name
    ticks_path = [p for p in list(data_path.glob("*")) if re.findall("ticks", str(p))][
        0
    ]

    df = pd.read_csv(ticks_path)
    print(f"{dir_name[:25]} - Total data shape: ", df.shape)

    df = df[df.pair.isin(pairs)]
    print(f"{dir_name[:25]} - Pairs data shape: ", df.shape)

    df = df.sort_values("write_time")
    df["write_timestamp"] = pd.to_datetime(df.write_time, unit="s")
    df["time_diff"] = df.write_timestamp.diff().dt.total_seconds()

    df = df.reset_index(drop=True)
    df = df.set_index("write_timestamp", drop=False)

    df["spread"] = (df.bid - df.ask).abs()

    df["collection_run"] = dir_name

    return df


def _load_and_add_to_dict(dir_name, pairs, dfs_dict):

    dfs_dict[dir_name] = _load_ticks(dir_name, pairs)


def drop_overlapping(df, print_prefix=""):

    min_times = df.groupby("collection_run")["write_timestamp"].min().sort_index()
    max_times = df.groupby("collection_run")["write_timestamp"].max().sort_index()

    for prev_idx, min_time in enumerate(min_times[1:]):

        if max_times[prev_idx] > min_time:

            prev_dir_name = min_times.index[prev_idx]

            subseq_run = df.collection_run == prev_dir_name
            overlapping = subseq_run & (df.write_time > min_time.timestamp())

            df = df[~overlapping]

            print(
                f"{print_prefix} - {prev_dir_name} - Dropped {overlapping.sum()} rows"
            )

    return df.copy()


def load_train_test(train_dirs, test_dirs):

    all_dirs = train_dirs + test_dirs

    with multiprocessing.Manager() as manager:

        dfs = manager.dict()

        n_dirs = len(all_dirs)
        with multiprocessing.Pool(processes=min(n_dirs, 6)) as pool:

            args = zip(all_dirs, [["ETH-USDT"]] * n_dirs, [dfs] * n_dirs)
            pool.starmap(_load_and_add_to_dict, args)

        dfs_train = [df for name, df in dfs.items() if name in train_dirs]
        dfs_test = [df for name, df in dfs.items() if name in test_dirs]

    df_train = pd.concat(dfs_train)
    df_test = pd.concat(dfs_test)

    df_train = drop_overlapping(df_train, "train")
    df_test = drop_overlapping(df_test, "test")

    return df_train, df_test


def _get_rolling_block(x, window_size):
    # https://stackoverflow.com/a/47483615/4909087
    # https://stackoverflow.com/a/46199050/4800652
    out_shape = (len(x) - (window_size - 1), window_size)
    out = np.lib.stride_tricks.as_strided(
        x, shape=out_shape, strides=(x.values.strides * 2)
    )
    return out


def blockify_data(x, n_history, n_ahead=0):

    blocks = _get_rolling_block(x, n_history + n_ahead)

    history = blocks[:, :n_history]
    price = history[:, -1].reshape(-1)

    assert history.shape[1] == n_history
    assert len(price) == history.shape[0]

    if n_ahead > 0:
        ahead = blocks[:, n_history:]
        assert ahead.shape[1] == n_ahead
        assert len(price) == ahead.shape[0]
    else:
        ahead = None

    return price, history, ahead

def create_targets(history, ahead, judge_buy, judge_sell):
    
    buy = judge_buy(history, ahead)        
    sell = judge_sell(history, ahead)
        
    targets = pd.Series(["hold"] * history.shape[0])
    targets[buy] = "buy"
    targets[sell] = "sell"
        
    return targets

def _chunk_stats(x, col_name, fn, left_idx, right_idx, **fn_kwargs):
    
    chunk = x[:, left_idx:right_idx]
    
    df = pd.DataFrame({
        col_name: fn(chunk, axis=1, **fn_kwargs)
    })
            
    return df

def _prep_chunk_args(mean, std, start, stop, prefix):
    
    if mean and std:
        raise Exception('Only one of mean or std can be True at same time')
    
    elif mean:
        fn_type = "mean"
        fn = np.mean
    
    elif std:
        
        fn_type = "std"
        fn = np.std
        
    else:
        raise Exception('Either mean or std should be True'
                       )
    col_name = f"{prefix}_{fn_type}_{start}_{stop}" 
    
    return col_name, fn

def chunker(x, prefix, size, skip_first=0, mean=False, std=False, **fn_kwargs):
        
    if std and size < 5:
        return None
    
    start, stop = skip_first, skip_first + size
    col_name, fn = _prep_chunk_args(mean, std, start, stop, prefix)
    
    left_idx = -(stop+1)
    right_idx = -(start+1) if start > 0 else None
    
    return _chunk_stats(x, col_name, fn, left_idx, right_idx, **fn_kwargs)
