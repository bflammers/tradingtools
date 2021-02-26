import logging
import re

logger = logging.getLogger(__name__)

import pandas as pd

from pathlib import Path
from cryptofeed.exchanges import Binance, Bitstamp, Kraken, Coinbase, Gemini

try:
    from ..datafeeding import DataFeed
    from ..datautils import (
        CSVWriter,
        create_results_dir,
        trades_columns,
        ticks_columns,
        nbbo_columns,
    )
except:
    from tradingtools.data.datafeeding import DataFeed
    from tradingtools.data.datautils import (
        CSVWriter,
        create_results_dir,
        trades_columns,
        ticks_columns,
        nbbo_columns,
    )


def binance_collect_ticks_trades_nbbo(results_dir_path: Path = "./data/collected/binance"):

    logger.info("Initializing new Datafeed")
    feed = DataFeed(exchange=Binance)

    logger.info("Creating results directory")
    results_dir = create_results_dir(results_dir_path)

    logger.info("Initialize CSVWriters")
    now = pd.Timestamp.now().strftime("%F_%H%M%S")
    ticks_writer = CSVWriter(
        path=results_dir / f"{now}_ticks.csv", columns=ticks_columns
    )
    trades_writer = CSVWriter(
        path=results_dir / f"{now}_trades.csv", columns=trades_columns
    )
    # nbbo_writer = CSVWriter(path=results_dir / f"{now}_nbbos.csv", columns=nbbo_columns)

    logger.info("Adding CSVWriters as consumers to DataFeed")
    feed.add_consumers(
        ticks_consumer=ticks_writer,
        trades_consumer=trades_writer, 
        nbbo_consumer=None)

    logger.info("Add instruments")
    binance_info = Binance.info()
    all_pairs = binance_info["pairs"]
    pairs = [pair for pair in all_pairs if re.findall("-USDT|-EUR|-USD", pair)]
    logger.info(f"{len(pairs)} instruments added")
    feed.add_instruments(pairs)

    # logger.info("Adding nbbo")
    # feed.add_nbbo(
    #     exchanges=[Bitstamp, Coinbase, Gemini, Kraken],
    #     instruments=["BTC-USD", "ETH-USD"]
    # )

    logger.info(f"Running.... writing ticks and trades to {results_dir}")
    feed.run()


if __name__ == "__main__":

    binance_collect_ticks_trades_nbbo()
