import re
import multiprocessing

import pandas as pd
import numpy as np

from pathlib import Path

from tqdm.notebook import tqdm_notebook as tqdm


#### Dataloading


def _load_and_add_to_dict(dir_name, dfs_dict, pairs=None):

    # Construct path
    if Path("./data/collected/binance").exists():
        binance_path = Path("data/collected/binance")
    else:
        binance_path = Path.cwd().parent / "data/collected/binance"
    data_path = binance_path / dir_name
    ticks_path = [p for p in data_path.glob("*") if re.findall("ticks", str(p))][0]

    # Read data
    df = pd.read_csv(ticks_path)
    print(f"{dir_name[:25]} - Total data shape: ", df.shape)

    # Filter pairs
    if pairs is not None:
        df = df[df.pair.isin(pairs)]
        print(f"{dir_name[:25]} - Pairs data shape: ", df.shape)

    # Add which collection run this data was collected in, for removing overlap later
    df["collection_run"] = dir_name

    # Add to dict
    dfs_dict[dir_name] = df


def load_train_val_test(
    train_dirs, val_dirs=None, test_dirs=None, pairs=["ETH-USDT"], n_pool=11
):

    all_dirs = train_dirs

    if val_dirs:
        all_dirs += val_dirs

    if test_dirs:
        all_dirs += test_dirs

    with multiprocessing.Manager() as manager:

        dfs = manager.dict()

        n_dirs = len(all_dirs)
        with multiprocessing.Pool(processes=min(n_dirs, n_pool)) as pool:

            args = zip(all_dirs, [dfs] * n_dirs, [pairs] * n_dirs)
            pool.starmap(_load_and_add_to_dict, args)

        # Train data
        df_train = pd.concat([df for name, df in dfs.items() if name in train_dirs])
        print(f"Train counts: \n{df_train.pair.value_counts()}")
        dups_train = df_train[["timestamp", "pair", "bid", "ask"]].duplicated()
        df_train = df_train[~dups_train]
        print(f"-- train dropped duplicates: {dups_train.sum()}")
        result = (df_train,)

        # Validation data
        if val_dirs:
            df_val = pd.concat([df for name, df in dfs.items() if name in val_dirs])
            print(f"Validation counts: \n{df_val.pair.value_counts()}")
            dups_val = df_val[["timestamp", "pair", "bid", "ask"]].duplicated()
            df_val = df_val[~dups_val]
            print(f"-- val dropped duplicates: {dups_train.sum()}")
            result += (df_val,)

        # Test data
        if test_dirs:
            df_test = pd.concat([df for name, df in dfs.items() if name in test_dirs])
            print(f"Test counts: \n{df_test.pair.value_counts()}")
            dups_test = df_test[["timestamp", "pair", "bid", "ask"]].duplicated()
            df_test = df_test[~dups_test]
            print(f"-- test dropped duplicates: {dups_test.sum()}")
            result += (df_test,)

    return result


#### Post-loading processing


def drop_overlapping(df):

    min_times = df.groupby("collection_run")["write_datetime"].min().sort_index()
    max_times = df.groupby("collection_run")["write_datetime"].max().sort_index()

    for prev_idx, min_time in enumerate(min_times[1:]):

        if max_times[prev_idx] > min_time:

            prev_dir_name = min_times.index[prev_idx]

            subseq_run = df.collection_run == prev_dir_name
            overlapping = subseq_run & (df.write_time > min_time.timestamp())

            df = df[~overlapping]

            print(f"{prev_dir_name} - Dropped {overlapping.sum()} rows due to overlap")

    return df.copy()


def get_rolling_block(x, window_size, same_size=False, center=False):
    len_decrease = window_size - 1
    out_shape = (len(x) - len_decrease, window_size)

    # Stride tricks for getting rolling window effect
    # https://stackoverflow.com/a/47483615/4909087
    # https://stackoverflow.com/a/46199050/4800652
    strides = x.values.strides if isinstance(x, pd.Series) else x.strides
    out = np.lib.stride_tricks.as_strided(x, shape=out_shape, strides=(strides * 2))

    # Prepend and / or post pend nans if output should be same size
    if same_size:

        # Prepend nans rows, size depends on whether the window should be centered
        prepend_len = int(np.ceil(len_decrease / 2)) if center else len_decrease
        prepend = np.full((prepend_len, window_size), np.nan)
        out = np.concatenate((prepend, out))

        if center:
            # Postpend nans, same size as prepend for even number of total rows to
            # pre/postpend, one less for odd number of total rows to pre/postpend
            postpend_len = int(np.floor(len_decrease / 2))
            postpend = np.full((postpend_len, window_size), np.nan)
            out = np.concatenate((out, postpend))

    return out


def blockify_data(x, n_history, n_ahead=0):

    # Transform price array (1-dimensional) into block array with windows (2-dimensional)
    window_size = n_history + n_ahead
    blocks = get_rolling_block(x, window_size, same_size=False)

    # Get index of each element in block array in original array x
    full_idx = np.arange(len(x))
    blocks_idx = get_rolling_block(full_idx, window_size, same_size=False)

    # Construct blocks (rolling windows)
    # history + index of the latest known value in history
    history = blocks[:, :n_history]
    latest_idx = blocks_idx[:, (n_history - 1)].reshape(-1)
    assert history.shape[1] == n_history

    # Ahead, set to None if n_ahead = 0
    if n_ahead > 0:
        ahead = blocks[:, n_history:]
        assert ahead.shape[1] == n_ahead
    else:
        ahead = None

    return history, latest_idx, ahead


def _pre_processing_pair(df_pair, n_history, n_ahead, price_col, ffill_limit=20):

    # Sort by time
    df_pair = df_pair.sort_values("timestamp")

    # Set write datetime as index
    df_pair["datetime"] = pd.to_datetime(df_pair["timestamp"], unit="s")
    df_pair = df_pair.set_index("datetime")

    # Drop duplicates
    df_pair = df_pair[~df_pair.index.duplicated()]

    # Price column, resample to 1 second frames
    price = df_pair[price_col].copy()
    price = price.resample("s").ffill(limit=ffill_limit)

    # Blockify
    history, latest_idx, ahead = blockify_data(price, n_history, n_ahead)
    original_price = price.iloc[latest_idx]

    # Calculate historical window mean and variance for standardization
    window_means = np.mean(history, axis=1)
    window_stds = np.std(history, axis=1)

    # Determine windows with zero variance
    # Determine windows with time gaps
    zero_var = window_stds < 1e-15
    time_gap = np.any(np.isnan(history), axis=1) | np.any(np.isnan(ahead), axis=1)
    gaps = zero_var | time_gap

    # Drop zero variance and timegaps windows
    original_price = original_price[~gaps]
    history = history[~gaps]
    latest_idx = latest_idx[~gaps]
    ahead = ahead[~gaps]
    window_means = window_means[~gaps]
    window_stds = window_stds[~gaps]

    print(f"-- Dropped {zero_var.sum()} rows due to (near-) zero variance")
    print(f"-- Dropped {time_gap.sum()} rows due to time gaps")

    # Standardize
    history -= window_means[:, None]
    history /= window_stds[:, None]
    ahead -= window_means[:, None]
    ahead /= window_stds[:, None]

    # Construct boolean column that indicates when a timegap/zerovar is about to happen
    upcoming_gap = np.append(gaps[1:], False)
    upcoming_gap = upcoming_gap[~gaps]

    # Data frame with info about latest price, latest idx, timegaps and dates
    # Not used in models (numphistory and ahead contain all information) but useful
    # for backtesting and visualization
    df_price_pair = pd.DataFrame(
        {
            "price": original_price,
            "standardized_price": history[:, -1],
            "upcoming_gap": upcoming_gap,
            "window_mean": window_means,
            "window_stds": window_stds,
        }
    )

    return df_price_pair, history, ahead


def pre_process(df, n_history=120, n_ahead=190, price_col="bid"):

    price_dfs = []
    history_arrays = []
    ahead_arrays = []

    for pair, df_pair in df.groupby("pair"):

        # Print pair
        print(f"Starting with: {pair}, shape: {df_pair.shape}")

        # Pre process per pair
        price_pair, history_pair, ahead_pair = _pre_processing_pair(
            df_pair, n_history, n_ahead, price_col
        )

        # Add coin to price pair dataframe
        price_pair["pair"] = pair

        # Append to holders
        price_dfs.append(price_pair)
        history_arrays.append(history_pair)
        ahead_arrays.append(ahead_pair)

    # Concatenate arrays and dataframes
    df_price = pd.concat(price_dfs)
    history = np.concatenate(history_arrays)
    ahead = np.concatenate(ahead_arrays)

    return df_price, history, ahead


#### Feature creation


def _chunk_stats(x, col_name, fn, left_idx, right_idx, **fn_kwargs):

    chunk = x[:, left_idx:right_idx]

    df = pd.DataFrame({col_name: fn(chunk, axis=1, **fn_kwargs)})

    return df


def _prep_chunk_args(mean, std, start, stop, prefix):

    if mean and std:
        raise Exception("Only one of mean or std can be True at same time")

    elif mean:
        fn_type = "mean"
        fn = np.mean

    elif std:

        fn_type = "std"
        fn = np.std

    else:
        raise Exception("Either mean or std should be True")
    col_name = f"{prefix}_{fn_type}_{start}_{stop}"

    return col_name, fn


def chunker(x, prefix, size, skip_first=0, mean=False, std=False, **fn_kwargs):

    if std and size < 5:
        return None

    start, stop = skip_first, skip_first + size
    col_name, fn = _prep_chunk_args(mean, std, start, stop, prefix)

    left_idx = -(stop + 1)
    right_idx = -(start + 1) if start > 0 else None

    return _chunk_stats(x, col_name, fn, left_idx, right_idx, **fn_kwargs)


def create_features(price, chunk_idx, n_first_diff=10):

    # Notes:
    # Quantiles and spread no positive effect

    first_diff = np.diff(price, axis=0)

    df_features = pd.DataFrame()

    for i in range(n_first_diff):
        df_features[f"first_diff_{i}"] = first_diff[:, -(i + 1)]

    for i, ci in tqdm(enumerate(chunk_idx), total=len(chunk_idx)):

        bid_incl_mean = chunker(price, "bid", size=ci, mean=True)
        fd_incl_mean = chunker(first_diff, "first_diff", size=ci, mean=True)
        bid_incl_std = chunker(price, "bid", size=ci, std=True)
        fd_incl_std = chunker(first_diff, "first_diff", size=ci, std=True)

        if i == 0:

            df_features = pd.concat(
                [df_features, bid_incl_mean, bid_incl_std, fd_incl_mean, fd_incl_std],
                axis=1,
            )

        else:

            prev_ci = chunk_idx[i - 1]
            size = ci - prev_ci

            bid_seq_mean = chunker(
                price, "bid", size=size, skip_first=prev_ci, mean=True
            )
            bid_seq_std = chunker(price, "bid", size=size, skip_first=prev_ci, std=True)
            fd_seq_mean = chunker(
                first_diff, "first_diff", size=size, skip_first=prev_ci, mean=True
            )
            fd_seq_std = chunker(
                first_diff, "first_diff", size=size, skip_first=prev_ci, std=True
            )

            df_features = pd.concat(
                [
                    df_features,
                    bid_incl_mean,
                    bid_incl_std,
                    fd_incl_mean,
                    fd_incl_std,
                    bid_seq_mean,
                    bid_seq_std,
                    fd_seq_mean,
                    fd_seq_std,
                ],
                axis=1,
            )

    # Demean by largest bid_mean inclusive column
    sel_cols = [x for x in list(df_features.columns) if re.findall("bid_mean", x)]
    demean_col = df_features[f"bid_mean_0_120"]
    df_features[sel_cols] = df_features[sel_cols].subtract(demean_col, axis=0)
    df_features = df_features.drop(columns=[f"bid_mean_0_120"])

    return df_features


if __name__ == "__main__":

    train_dirs = ["2021-04-03_235900_9f1b26c870364ee9ac41f4bbf522cb69"]

    val_dirs = None

    test_dirs = None

    pairs = ["BTC-USDT", "ETH-USDT", "XRP-BNB", "ADA-BNB", "DOT-BNB"]
    (train_data,) = load_train_val_test(train_dirs, val_dirs, test_dirs, pairs)

    all_prices, all_history, all_ahead = pre_process(train_data, 120, 190)
