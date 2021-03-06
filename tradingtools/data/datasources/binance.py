import logging
import re

from itertools import product


logger = logging.getLogger(__name__)

import pandas as pd

from pathlib import Path
from cryptofeed.exchanges import Binance

try:
    from ..datafeeding import DataFeed
    from ..datautils import CSVWriter, create_results_dir, trades_columns, ticks_columns
except:
    from tradingtools.data.datafeeding import DataFeed
    from tradingtools.data.datautils import (
        CSVWriter,
        create_results_dir,
        trades_columns,
        ticks_columns,
    )


def binance_collect(
    results_dir_path: Path = "./data/testing/",
    ticks=True,
    trades=True,
    coin_symbols=["BTC", "ETH"],
    base_symbols=["USDT", "BNB"]
):

    logger.info("Initializing new Datafeed")
    feed = DataFeed(exchange=Binance)

    logger.info("Creating results directory")
    results_dir = create_results_dir(results_dir_path)

    logger.info("Initialize CSVWriters")
    now = pd.Timestamp.now().strftime("%F_%H%M%S")

    if ticks:
        ticks_writer = CSVWriter(
            path=results_dir / f"{now}_ticks.csv", columns=ticks_columns
        )
    else:
        ticks_writer = None

    if trades:
        trades_writer = CSVWriter(
            path=results_dir / f"{now}_trades.csv", columns=trades_columns
        )
    else:
        trades_writer = None

    logger.info("Adding CSVWriters as consumers to DataFeed")
    feed.add_consumers(
        ticks_consumer=ticks_writer, trades_consumer=trades_writer, nbbo_consumer=None
    )

    logger.info("Add instruments")

    # Get Binance instruments
    binance_info = Binance.info()
    all_pairs = binance_info["symbols"]

    # Find instruments that match pattern
    pairs_input = ["-".join(pair) for pair in product(coin_symbols, base_symbols)]
    pairs_regex = "|".join(pairs_input)
    pairs = [pair for pair in all_pairs if re.findall(pairs_regex, pair)]
    logger.info(f"Starting collection on Binance for pairs: {pairs}")

    pairs_missing = set(pairs_input) - set(pairs)
    if len(pairs) > 0:
        logger.warning(f"Input pairs not available on binance: {pairs_missing}")
    
    # Add instruments to feed
    feed.add_instruments(pairs)
    logger.info(f"{len(pairs)} instruments added")

    logger.info(f"Running.... writing data to {results_dir}")
    feed.run()


if __name__ == "__main__":

    binance_collect()
