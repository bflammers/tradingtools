import logging

logger = logging.getLogger(__name__)

import pandas as pd

from pathlib import Path
from cryptofeed.exchanges import Deribit

try:
    from ..datafeeding import OptionsDataFeed
    from ..datautils import CSVWriter, create_results_dir, trades_columns, ticks_columns
except:
    from tradingtools.data.datafeeding import OptionsDataFeed
    from tradingtools.data.datautils import (
        CSVWriter,
        create_results_dir,
        trades_columns,
        ticks_columns,
        nbbo_columns,
    )


def deribit_collect_ticks_trades(results_dir_path: Path = "./data/collected/Deribit"):

    logger.info("Initializing new OptionsCollectionFeed")
    ocf = OptionsDataFeed(exchange=Deribit)

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
    nbbo_writer = CSVWriter(path=results_dir / f"{now}_nbbos.csv", columns=nbbo_columns)

    logger.info("Adding CSVWriters as consumers to DataFeed")
    ocf.add_consumers(
        ticks_consumer=ticks_writer,
        trades_consumer=trades_writer,
        nbbo_consumer=nbbo_writer,
    )

    logger.info("Syncing instruments")
    ocf.sync_instruments()

    logger.info("Adding nbbo")
    ocf.add_nbbo()

    logger.info(f"Running.... writing ticks and trades to {results_dir}")
    ocf.run()


if __name__ == "__main__":

    parent_path = Path("/media/bart/Bart_500GB/cryptodata/collected/deribit")
    deribit_collect_ticks_trades(parent_path)
