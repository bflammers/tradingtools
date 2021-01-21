from uuid import uuid4
from pathlib import Path
from threading import Thread
from queue import Queue

import pandas as pd

from cryptofeed import FeedHandler
from cryptofeed.callback import TickerCallback, TradeCallback
from cryptofeed.defines import TICKER, TRADES
from cryptofeed.exchanges import Deribit
from cryptofeed.feed import Feed

try:
    from .utils import CSVWriter
except:
    from utils import CSVWriter


class CollectionFeed:
    fh: FeedHandler
    exchange: Feed
    running: bool

    def __init__(
        self,
        exchange,
        collect_ticker: bool = True,
        collect_trade: bool = True,
        parent_dir: str = ".",
    ) -> None:
        super().__init__()
        self.exchange = exchange
        self.fh = FeedHandler()
        self.running = False
        self.current_instruments = set()

        # Initialize writer attributes
        self._results_dir = None
        self.ticker_q = None
        self.ticker_writer = None
        self.trade_q = None
        self.trade_writer = None
        self._writers_running = False

        self.collect_ticker = collect_ticker
        self.collect_trade = collect_trade

        # Create directory for results
        now = pd.Timestamp.now().strftime("%F_%T")
        ts_uuid = f"{now}_{uuid4().hex}"
        self._results_dir = (Path(parent_dir) / ts_uuid).absolute()
        self._results_dir.mkdir(parents=True, exist_ok=True)

    def start_writers(self) -> None:

        now = pd.Timestamp.now().strftime("%F_%T")

        if self.collect_ticker:
            self.ticker_q = Queue()
            self.ticker_writer = CSVWriter(
                path=self._results_dir / f"{now}_ticker.csv",
                columns=[
                    "feed",
                    "timestamp",
                    "receipt_timestamp",
                    "pair",
                    "bid",
                    "ask",
                ],
            )
            self.ticker_writer.start_worker(self.ticker_q)

        if self.collect_trade:
            self.trade_q = Queue()
            self.trade_writer = CSVWriter(
                path=self._results_dir / f"{now}_trades.csv",
                columns=[
                    "feed",
                    "timestamp",
                    "receipt_timestamp",
                    "pair",
                    "order_id",
                    "side",
                    "amount",
                    "price",
                ],
            )
            self.trade_writer.start_worker(self.trade_q)

        self._writers_running = True

    def add_instruments(self):
        raise NotImplementedError

    def run(self):

        if not self._writers_running:
            Exception(
                "[CollectionFeed.run] writers not started, call start_writers() first"
            )

        self.fh.run()
        self.running = True

    async def ticker_callback(self, feed, pair, bid, ask, timestamp, receipt_timestamp):
        # print(f"Ticker q len: {self.ticker_q.qsize()}")
        self.ticker_q.put(
            {
                "feed": feed,
                "timestamp": timestamp,
                "receipt_timestamp": receipt_timestamp,
                "pair": pair,
                "bid": bid,
                "ask": ask,
            }
        )

    async def trade_callback(
        self, feed, pair, order_id, timestamp, side, amount, price, receipt_timestamp
    ):
        # print(f"Trade q len: {self.trade_q.qsize()}")
        self.trade_q.put(
            {
                "feed": feed,
                "timestamp": timestamp,
                "receipt_timestamp": receipt_timestamp,
                "pair": pair,
                "order_id": order_id,
                "side": side,
                "amount": amount,
                "price": price,
            }
        )


class OptionsCollectionFeed(CollectionFeed):
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
                    TRADES: TradeCallback(self.trade_callback),
                },
            )
        )

        # Update current instruments
        self.current_instruments |= new_instruments

    def run(self):
        self.fh.run()
        self.running = True


if __name__ == "__main__":

    print("Initializing new OptionsCollectionFeed")
    ocf = OptionsCollectionFeed(exchange=Deribit, parent_dir="./data/collected/Deribit")
    
    print("Syncing instruments")
    ocf.sync_instruments()
    
    print("Starting CSV file writer threads")
    ocf.start_writers()

    print(f"Running.... writing ticks and trades to {ocf._results_dir}")
    ocf.run()
