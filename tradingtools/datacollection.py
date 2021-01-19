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


async def trade(
    feed, pair, order_id, timestamp, side, amount, price, receipt_timestamp
):
    print(
        f"TRADE >> Timestamp: {timestamp} Feed: {feed} Pair: {pair} ID: {order_id} Side: {side} Amount: {amount} Price: {price}"
    )


async def ticker(feed, pair, bid, ask, timestamp, receipt_timestamp):
    print(
        f"TICKER >> Timestamp: {timestamp} Feed: {feed} Pair: {pair} Bid: {bid} Ask: {ask}"
    )


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

        self.collect_ticker = collect_ticker
        self.collect_trade = collect_trade

        # Create directory for results
        now = pd.Timestamp.now().strftime("%Y%m%d_%H%M")
        ts_uuid = f"{now}_{uuid4().hex}"
        self._results_dir = (Path(parent_dir) / ts_uuid).absolute()
        self._results_dir.mkdir(parents=True, exist_ok=True)

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
            self._start_writer_worker(self.ticker_q, self.ticker_writer)

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
            self._start_writer_worker(self.trade_q, self.trade_writer)

    def add_instruments(self):
        raise NotImplementedError

    def run(self):
        self.fh.run()

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

    def _writer_worker(self, q: Queue, writer: CSVWriter) -> None:
        while True:
            row_dict = q.get()
            writer.append(row_dict)
            q.task_done()

    def _start_writer_worker(self, q: Queue, writer: CSVWriter):
        thr = Thread(target=self._writer_worker, args=(q, writer))
        thr.daemon = True
        thr.start()


class OptionsCollectionFeed(CollectionFeed):
    def __init__(
        self,
        exchange,
        collect_ticker: bool = True,
        collect_trade: bool = True,
        parent_dir: str = ".",
    ) -> None:
        super().__init__(exchange, collect_ticker, collect_trade, parent_dir)

    def add_instruments(self):

        exchange_instruments = self.exchange.get_instruments_info()

        df_instruments = pd.DataFrame(exchange_instruments["result"])
        df_instruments["created"] = pd.to_datetime(df_instruments["created"])
        df_instruments["expiration"] = pd.to_datetime(
            df_instruments["expiration"], errors="coerce"
        )
        df_instruments["days_total"] = (
            df_instruments.expiration - df_instruments.created
        ).dt.days
        df_instruments["days_remaining"] = (
            df_instruments.expiration - pd.Timestamp.utcnow()
        ).dt.days

        df_selected = df_instruments[
            (df_instruments.baseCurrency == "BTC")
        ].reset_index(drop=True)

        instruments = df_selected.instrumentName.tolist()

        config = {TRADES: instruments, TICKER: instruments}
        self.fh.add_feed(
            self.exchange(
                config=config,
                callbacks={
                    TICKER: TickerCallback(self.ticker_callback),
                    TRADES: TradeCallback(self.trade_callback),
                },
            )
        )

        if self.running:
            self.fh.run()


if __name__ == "__main__":

    ocf = OptionsCollectionFeed(exchange=Deribit)
    ocf.add_instruments()
    ocf.run()
