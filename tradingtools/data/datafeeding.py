from typing import Callable
from uuid import uuid4
from pathlib import Path

import pandas as pd

from cryptofeed import FeedHandler
from cryptofeed.callback import TickerCallback, TradeCallback
from cryptofeed.defines import TICKER, TRADES
from cryptofeed.exchanges import Deribit
from cryptofeed.feed import Feed

try:
    from .datahandling import ThreadStream
    from .datautils import make_ticker_callback, make_trade_callback
except ImportError:
    from tradingtools.data.datahandling import ThreadStream
    from tradingtools.data.datautils import make_ticker_callback, make_trade_callback


class DataFeed:
    fh: FeedHandler
    exchange: Feed
    running: bool
    ticker_callback: Callable
    trades_callback: Callable
    _consumers_running: bool = False

    def __init__(
        self,
        exchange,
    ) -> None:
        super().__init__()
        self.exchange = exchange
        self.fh = FeedHandler()
        self.running = False
        self.current_instruments = set()

    def add_consumers(
        self, ticks_consumer: ThreadStream, trades_consumer: ThreadStream
    ) -> None:

        self.ticker_callback = make_ticker_callback(ticks_consumer.add_to_q)
        self.trades_callback = make_trade_callback(trades_consumer.add_to_q)
        self._consumers_running = True

    def add_instruments(self):
        raise NotImplementedError

    def run(self):

        if not self._consumers_running:
            Exception("[DataFeed] consumers not started, call first")

        self.fh.run()
        self.running = True


class OptionsDataFeed(DataFeed):
    def sync_instruments(
        self, base_symbol: str = None, max_expiration_days: int = None
    ):

        # Fetch instruments and make DataFrame
        exchange_instruments = self.exchange.get_instruments_info()
        df_instruments = pd.DataFrame(exchange_instruments["result"])

        # Convert time columns to datetime
        df_instruments["created"] = pd.to_datetime(
            df_instruments["created"], errors="coerce"
        )
        df_instruments["expiration"] = pd.to_datetime(
            df_instruments["expiration"], errors="coerce"
        )

        # Calculate time differences in days
        time_total = df_instruments["expiration"] - df_instruments["created"]
        df_instruments["days_total"] = time_total.dt.days
        time_remaining = df_instruments["expiration"] - pd.Timestamp.utcnow()
        df_instruments["days_remaining"] = time_remaining.dt.days

        if base_symbol:
            df_instruments = df_instruments[
                (df_instruments["baseCurrency"] == base_symbol)
            ].reset_index(drop=True)

        if max_expiration_days:
            df_instruments = df_instruments[
                (df_instruments["days_remaining"] <= max_expiration_days)
            ].reset_index(drop=True)

        # Get instruments and remove duplicates
        instruments = list(set(df_instruments["instrumentName"].tolist()))

        # Add the instruments to the feed
        self.add_instruments(instruments)

    def add_instruments(self, instruments: list):

        # Determine new instruments
        new_instruments = set(instruments) - self.current_instruments

        # Add instruments to feed
        config = {TRADES: list(new_instruments), TICKER: list(new_instruments)}
        self.fh.add_feed(
            self.exchange(
                config=config,
                callbacks={
                    TICKER: TickerCallback(self.ticker_callback),
                    TRADES: TradeCallback(self.trades_callback),
                },
            )
        )

        # Update current instruments
        self.current_instruments |= new_instruments


if __name__ == "__main__":

    print("Initializing new OptionsCollectionFeed")
    ocf = OptionsDataFeed(exchange=Deribit, parent_dir="./data/collected/Deribit")

    ticks_handler = ThreadStream()
    ticks_handler.add_consumer(
        lambda x: print('Ticks: ', len(x), flush=True), interval_time=1, batched=True
    )

    trades_handler = ThreadStream()
    trades_handler.add_consumer(
        lambda x: print('Trades: ', len(x), flush=True), interval_time=1, batched=True
    )

    ocf.add_consumers(ticks_consumer=ticks_handler, trades_consumer=trades_handler)

    print("Syncing instruments")
    ocf.sync_instruments()

    print(f"Running.... writing ticks and trades to {ocf._results_dir}")
    ocf.run()
