import multiprocessing
import logging

import pandas as pd
import numpy as np

from tqdm.notebook import tqdm_notebook as tqdm

logger = logging.getLogger(__name__)


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


def _log_perc_affected(mask_affected, message, logging_type="INFO"):
    n = sum(mask_affected)
    perc = round(n / len(mask_affected) * 100, 2)
    if logging_type == "INFO":
        logger.info(f"{message} - affected {n} rows ({perc}%)")
    elif logging_type == "WARNING":
        logger.warning(f"{message} - affected {n} rows ({perc}%)")
    elif logging_type == "ERROR":
        logger.error(f"{message} - affected {n} rows ({perc}%)")
    else:
        raise Exception(
            f"logging_type {logging_type}, not one of [INFO, WARNING, ERROR]"
        )


class PreProcessor:
    def __init__(
        self, n_history=600, n_ahead=240, ffill_limit=20, price_col="bid", n_pool=11
    ) -> None:
        self.n_history = n_history
        self.n_ahead = n_ahead
        self.ffill_limit = ffill_limit
        self.price_col = price_col
        self.n_pool = n_pool

        self.price_dfs = None
        self.history_arrays = None
        self.ahead_arrays = None

        if n_history < 50:
            logger.warning(
                f"[pre-process] n_history argument is small ({n_history}), might lead to unstable standardization"
            )

    @staticmethod
    def blockify_data(x, n_history, n_ahead):

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

    def _pre_processing_pair(self, pair, df_pair):

        # Drop duplicate timestamps
        dup_timestamp = df_pair["timestamp"].duplicated()
        df_pair = df_pair[~dup_timestamp].copy()
        _log_perc_affected(dup_timestamp, f"[{pair}] Dropping rows with duplicate timestamps")

        # Sort by time
        df_pair = df_pair.sort_values("timestamp")

        # Set write datetime as index
        df_pair["datetime"] = pd.to_datetime(df_pair["timestamp"], unit="s")
        df_pair = df_pair.set_index("datetime")

        # Drop duplicates
        df_pair = df_pair[~df_pair.index.duplicated()]

        # Price column, resample to 1 second frames
        price = df_pair[self.price_col].copy()
        price = price.resample("s").ffill(limit=self.ffill_limit)

        # Blockify
        history, latest_idx, ahead = self.blockify_data(
            price, self.n_history, self.n_ahead
        )
        original_price = price.iloc[latest_idx]

        # Calculate historical window mean and variance for standardization
        window_means = np.mean(history, axis=1)
        window_stds = np.std(history, axis=1)

        # Set zero var standard deviations to 1 to prevent division by 0
        zero_var = window_stds < 1e-15
        window_stds[zero_var] = 1
        _log_perc_affected(zero_var, f"[{pair}] Not scaling zero variance windows")

        # Determine windows with zero variance
        # Determine windows with time gaps
        gaps = np.any(np.isnan(history), axis=1) | np.any(np.isnan(ahead), axis=1)
        _log_perc_affected(gaps, f"[{pair}] Dropping windows with timegaps")

        # Drop timegaps windows
        original_price = original_price[~gaps]
        history = history[~gaps]
        latest_idx = latest_idx[~gaps]
        ahead = ahead[~gaps]
        window_means = window_means[~gaps]
        window_stds = window_stds[~gaps]

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

    def _pre_process_pair_wrapper(self, pair, df_pair):

        # Log pair
        logger.info(f"[{pair}] shape: {df_pair.shape}")

        if df_pair.shape[0] < self.n_history + self.n_ahead:
            logger.warning(f"[{pair}] Only {df_pair.shape[0]} rows, skipping {pair}")
            return

        # Pre process per pair
        price_pair, history_pair, ahead_pair = self._pre_processing_pair(
            pair=pair, df_pair=df_pair
        )

        # Add pair to dataframe
        price_pair["pair"] = pair

        # Add to multiprocessing dicts
        self.price_dfs[pair] = price_pair
        self.history_arrays[pair] = history_pair
        self.ahead_arrays[pair] = ahead_pair

    def pre_process(self, df):

        with multiprocessing.Manager() as manager:

            # Create multiprocessing dicts
            self.price_dfs = manager.dict()
            self.history_arrays = manager.dict()
            self.ahead_arrays = manager.dict()

            dfg = df.groupby("pair")
            n_pairs = len(dfg)

            with multiprocessing.Pool(processes=min(n_pairs, self.n_pool)) as pool:

                pool.starmap(self._pre_process_pair_wrapper, dfg)

            # Concatenate arrays and dataframes
            df_price = pd.concat(self.price_dfs.values())
            history = np.concatenate(self.history_arrays.values())
            ahead = np.concatenate(self.ahead_arrays.values())

        return df_price, history, ahead


###############################################################
#### Feature creation
###############################################################


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

