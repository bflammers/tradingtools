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
    from ..datahandling import BatchedTickHandler
except:
    from tradingtools.datahandling import BatchedTickHandler


class CollectionFeed:
    fh: FeedHandler
    exchange: Feed
    running: bool

    def __init__(
        self,
        exchange
    ) -> None:
        super().__init__()
        self.exchange = exchange
        self.fh = FeedHandler()
        self.running = False
        self.current_instruments = set()

        self.tick_handler = BatchedTickHandler(lambda x: print(f"Tick: {len(x)}"), 1)
        self.trade_handler = BatchedTickHandler(lambda x: print(f"Trade: {len(x)}"), 1)

    async def ticker_callback(self, feed, pair, bid, ask, timestamp, receipt_timestamp):
        # print(f"Ticker q len: {self.ticker_q.qsize()}")
        self.tick_handler.process(
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
        self.trade_handler.process(
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
    ocf = CollectionFeed(exchange=Deribit)
    
    print("Syncing instruments")
    # ocf.sync_instruments()
    ocf.add_instruments(['ETH-PERPETUAL'])

    print(f"Running.... ")
    ocf.run()
