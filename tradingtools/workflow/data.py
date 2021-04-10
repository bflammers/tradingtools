import re
import multiprocessing

import pandas as pd
import numpy as np

from pathlib import Path

from tqdm.notebook import tqdm_notebook as tqdm


#### Dataloading


def _load_and_add_to_dict(dir_name, pairs, dfs_dict):

    # Construct path
    binance_path = Path.cwd().parent / "data/collected/binance"
    data_path = binance_path / dir_name
    ticks_path = [p for p in list(data_path.glob("*")) if re.findall("ticks", str(p))][
        0
    ]

    # Read data
    df = pd.read_csv(ticks_path)
    print(f"{dir_name[:25]} - Total data shape: ", df.shape)

    # Filter pairs
    df = df[df.pair.isin(pairs)]
    print(f"{dir_name[:25]} - Pairs data shape: ", df.shape)

    # Add which collection run this data was collected in, for removing overlap later
    df["collection_run"] = dir_name

    # Add to dict
    dfs_dict[dir_name] = df


def load_train_val_test(train_dirs, val_dirs=None, test_dirs=None, n_pool=11):

    all_dirs = train_dirs

    if val_dirs:
        all_dirs += val_dirs

    if test_dirs:
        all_dirs += test_dirs

    with multiprocessing.Manager() as manager:

        dfs = manager.dict()

        n_dirs = len(all_dirs)
        with multiprocessing.Pool(processes=min(n_dirs, n_pool)) as pool:

            args = zip(all_dirs, [["ETH-USDT"]] * n_dirs, [dfs] * n_dirs)
            pool.starmap(_load_and_add_to_dict, args)

        # Train data
        dfs_train = [df for name, df in dfs.items() if name in train_dirs]
        df_train = pd.concat(dfs_train)
        result = (df_train,)

        # Validation data
        if val_dirs:
            dfs_val = [df for name, df in dfs.items() if name in val_dirs]
            df_val = pd.concat(dfs_val)
            result += (df_val,)

        # Test data
        if test_dirs:
            dfs_test = [df for name, df in dfs.items() if name in test_dirs]
            df_test = pd.concat(dfs_test)
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


def _get_rolling_block(x, window_size):
    # https://stackoverflow.com/a/47483615/4909087
    # https://stackoverflow.com/a/46199050/4800652
    out_shape = (len(x) - (window_size - 1), window_size)
    out = np.lib.stride_tricks.as_strided(
        x, shape=out_shape, strides=(x.values.strides * 2)
    )
    return out


def blockify_data(x, n_history, n_ahead=0, split=True):

    blocks = _get_rolling_block(x, n_history + n_ahead)

    if not split:
        return blocks

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


def pre_process(df, n_history=120, n_ahead=240, price_col="bid"):

    # Sort by time
    df = df.sort_values("timestamp")

    # Add columns
    df["datetime"] = pd.to_datetime(df.timestamp, unit="s")
    df["write_datetime"] = pd.to_datetime(df.write_time, unit="s")
    df["time_diff"] = df.datetime.diff().dt.total_seconds()
    df["spread"] = (df.bid - df.ask).abs()

    # Set write datetime as index
    df = df.reset_index(drop=True)
    df = df.set_index("datetime", drop=True)

    # Drop rows with overlapping ticks
    # around mid-night when new job takes over there is a period of
    # time when the data is collected by two result writers
    df = drop_overlapping(df)

    # Blockify
    price, history, ahead = blockify_data(df[price_col], n_history, n_ahead)

    # Drop timegaps
    time_diff_blocks = blockify_data(df["time_diff"], n_history, n_ahead, split=False)
    time_gap_elements = (time_diff_blocks < 0.2) | (2 < time_diff_blocks)
    time_gap_blocks = np.any(time_gap_elements, axis=1)
    price, history, ahead = (
        price[~time_gap_blocks],
        history[~time_gap_blocks],
        ahead[~time_gap_blocks],
    )
    print(f"Dropped {time_gap_blocks.sum()} rows due to time gaps")

    return price, history, ahead, time_gap_blocks


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

    train_dirs = [
        "2021-04-03_235900_9f1b26c870364ee9ac41f4bbf522cb69",
        "2021-04-04_235900_d08a787886b846359231e418aa2654f4",
        "2021-04-05_235900_d633b719195c4991a040ef5d3a610f22",
        "2021-04-06_235900_864be414a4e04f80bba38dcef7d46053",
        "2021-04-07_235901_4c7e8300c7de48a285ca8816fc7bf6c5",
    ]

    val_dirs = None

    test_dirs = None

    train_data, val_data, test_data = load_train_val_test(
        train_dirs, val_dirs, test_dirs
    )
