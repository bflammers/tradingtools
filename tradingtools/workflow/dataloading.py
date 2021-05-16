
import multiprocessing
import logging

import pandas as pd
import numpy as np

from datetime import datetime
from pathlib import Path

from tqdm.notebook import tqdm_notebook as tqdm

logger = logging.getLogger(__name__)


#### Dataloading


class DataLoader:
    def __init__(
        self, parent_dir="./data/collected/binance", max_parents=3, n_pool=11
    ) -> None:
        self.paths = self.get_collection_paths(parent_dir, max_parents)
        self.n_pool = n_pool

    @staticmethod
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

    @staticmethod
    def _load_and_add_to_dict(path, dfs_dict, pairs=None):

        # Read data
        df = pd.read_csv(path)
        logger.info(f"{path.name} - Total data shape: {df.shape}")

        # Filter pairs
        if pairs is not None:
            df = df[df.pair.isin(pairs)]
            logger.info(f"{path.name} - Pairs data shape: {df.shape}")

        # Add which collection run this data was collected in, for removing overlap later
        df["collection_run"] = path.name

        # Add to dict
        dfs_dict[path.name] = df

    def load_paths(self, paths, pairs=None):

        path_names = [p.name for p in paths]
        if pairs:
            logger.info(
                f"Loading {len(pairs)} accross {len(paths)} paths -- pairs: {pairs} -- paths: {path_names}"
            )
        else:
            logger.info(
                f"Loading all pairs accross {len(paths)} paths -- paths: {path_names}"
            )

        with multiprocessing.Manager() as manager:

            dfs = manager.dict()

            n_dirs = len(paths)
            with multiprocessing.Pool(processes=min(n_dirs, self.n_pool)) as pool:

                args = zip(paths, [dfs] * n_dirs, [pairs] * n_dirs)
                pool.starmap(self._load_and_add_to_dict, args)

            # Train data
            df = pd.concat(dfs.values())
            logger.info(f"Train counts: \n{df.pair.value_counts()}")
            dups_train = df[["timestamp", "pair", "bid", "ask"]].duplicated()
            df = df[~dups_train]
            logger.info(f"-- train dropped duplicates: {dups_train.sum()}")

        return df

    def load_all(self, pairs=None):

        return self.load_paths(self.paths, pairs)

    def load_latest_n(self, n, pairs=None):
        paths = self.paths[-n:]
        return self.load_paths(paths, pairs)

    def load_by_date(self, min_datetime=None, max_datetime=None, pairs=None):

        # Select paths to load
        paths = np.array(self.paths)
        dates = np.array(
            [datetime.strptime(p.name[:17], "%Y-%m-%d_%H%M%S") for p in paths]
        )

        if min_datetime is None and max_datetime is None:
            logger.warning("No datetimes provided, loading everything")
            self.load_all(pairs)

        if min_datetime:
            paths = paths[dates >= min_datetime]
            
        if max_datetime:
            paths = paths[dates < max_datetime]

        return self.load_paths(paths, pairs)


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

            logger.info(
                f"{prev_dir_name} - Dropped {overlapping.sum()} rows due to overlap"
            )

    return df.copy()




if __name__ == "__main__":

    pass
