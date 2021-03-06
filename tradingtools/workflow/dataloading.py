import multiprocessing
import logging

import pandas as pd
import numpy as np

from datetime import datetime
from pathlib import Path

from tqdm.notebook import tqdm_notebook as tqdm
from tradingtools.data.datautils import threadsafe_generator

logger = logging.getLogger(__name__)


def get_collection_paths(parent_dir, max_parents=3):

    cd = Path.cwd()

    for i in range(max_parents + 1):

        data_path = cd / parent_dir

        if data_path.exists():
            logger.info(f"Data path found: {data_path}")
            break

        else:
            if i == max_parents:
                raise Exception(f"Data path not found in latest parent: {cd}")

            logger.info(f"Data path not found in {cd}, going to parent")
            cd = cd.parent

    # List data directories
    collection_paths = sorted(data_path.glob("*"))

    return collection_paths


class DataLoader:
    def __init__(
        self,
        parent_dir="./data/collected/binance",
        pairs=None,
        max_parents=3,
        n_pool=11,
        debug=False,
    ) -> None:
        self.paths = get_collection_paths(parent_dir, max_parents)
        self.pairs = pairs
        self.n_pool = n_pool
        self.debug = debug
        self.read_csv_kwargs = dict()

        if debug:
            self.read_csv_kwargs = {"nrows": 3000000}

            if pairs is None:
                self.pairs = ["BTC-USDT", "ETH-USDT", "XRP-BNB", "ADA-BNB"]

    @staticmethod
    def _filter_pairs_duplicates(df, pairs=None):

        # Filter pairs
        if pairs is not None:
            df = df[df["pair"].isin(pairs)]

        # Dropping duplicates
        dups = df[["timestamp", "pair", "bid", "ask"]].duplicated()
        if any(dups):
            logger.info(f"Dropping {sum(dups)} duplicates")
            df = df[~dups].copy()

        return df

    @staticmethod
    def _log_paths_pairs(paths, pairs=None):

        path_names = [p.name for p in paths]
        if pairs:
            logger.info(
                f"Loading {len(pairs)} accross {len(paths)} paths -- pairs: {pairs} -- paths: {path_names}"
            )
        else:
            logger.info(
                f"Loading all pairs accross {len(paths)} paths -- paths: {path_names}"
            )

    def exclude_by_date(self, max_date):
        paths_before = set(self.paths)
        self.paths = [p for p in self.paths if self.parse_datetime_path(p) >= max_date]
        excluded = sorted(list(paths_before - set(self.paths)))
        logger.warning(f"Excluded {len(excluded)} paths: {excluded}")

    def _load_and_add_to_dict(self, path, dfs_dict):

        # Read data
        df = pd.read_csv(path, **self.read_csv_kwargs)
        logger.info(f"{path.name} - Total data shape: {df.shape}")

        # Filter pairs
        df = self._filter_pairs_duplicates(df, self.pairs)
        logger.info(f"{path.name} - Pairs data shape: {df.shape}")

        # Add which collection run this data was collected in, for removing overlap later
        df["collection_run"] = path.name

        # Add to dict
        dfs_dict[path.name] = df

    def load_paths(self, paths):

        # Log total paths and pairs
        self._log_paths_pairs(paths, self.pairs)

        if self.debug:

            dfs = dict()

            logger.info("debug=True, running single threaded")
            for path in paths:
                self._load_and_add_to_dict(path, dfs)

            # Concat separate dfs
            df = pd.concat(dfs.values())

        else:

            with multiprocessing.Manager() as manager:

                dfs = manager.dict()

                n_dirs = len(paths)
                with multiprocessing.Pool(processes=min(n_dirs, self.n_pool)) as pool:

                    args = zip(paths, [dfs] * n_dirs)
                    pool.starmap(self._load_and_add_to_dict, args)

                    # Concat separate dfs
                    df = pd.concat(dfs.values())

        # Log value counts
        logger.info(f"Train counts: \n{df.pair.value_counts()}")

        # Drop duplicates
        dups_train = df[["timestamp", "pair", "bid", "ask"]].duplicated()
        df = df[~dups_train]
        logger.info(f"-- train dropped duplicates: {dups_train.sum()}")

        return df

    @staticmethod
    def parse_datetime_path(path):
        return datetime.strptime(path.name[:17], "%Y-%m-%d_%H%M%S")

    @threadsafe_generator
    def load_iterative(self, paths, n_next_append=1000000):

        # Log total paths and pairs
        self._log_paths_pairs(paths, self.pairs)

        # Reading initial file"
        logger.info(f"Reading initial file (1/{len(paths)}): {paths[0].name}")
        df_curr = pd.read_csv(paths[0], **self.read_csv_kwargs)

        for i, path in enumerate(paths[1:]):

            # Load next data file and append
            logger.info(f"Reading next file ({i + 2}/{len(paths)}): {path.name}")
            df_next = pd.read_csv(path, **self.read_csv_kwargs)
            df = pd.concat([df_curr, df_next.head(n_next_append)])

            # Filter not-selected pairs and duplicates
            df = self._filter_pairs_duplicates(df, self.pairs)

            # Set current data file
            df_curr = df_next.copy()

            yield df

        df_curr = self._filter_pairs_duplicates(df_curr, self.pairs)
        logger.info(f"Yielding last file (without head of next one appended)")

        yield df_curr

    def load_all(self, iterator=False):

        if iterator:
            return self.load_iterative(self.paths)

        return self.load_paths(self.paths)

    def load_latest_n(self, n, iterator=False):

        # Select the n latest paths
        paths = self.paths[-n:]

        if iterator:
            return self.load_iterative(paths)

        return self.load_paths(paths)

    def load_by_date(
        self,
        min_datetime=None,
        max_datetime=None,
        iterator=False,
    ):

        # Select paths to load
        paths = np.array(self.paths)
        dates = np.array(
            [datetime.strptime(p.name[:17], "%Y-%m-%d_%H%M%S") for p in paths]
        )

        if min_datetime is None and max_datetime is None:
            logger.warning("No datetimes provided, loading everything")
            return self.load_all(iterator=iterator)

        if min_datetime:
            paths = paths[dates >= min_datetime]

        if max_datetime:
            paths = paths[dates < max_datetime]

        if iterator:
            return self.load_iterative(paths)

        return self.load_paths(paths)


if __name__ == "__main__":

    dl = DataLoader(debug=True)
    train_data = dl.load_latest_n(5)
    print(train_data)
