from typing import Callable, List
from tradingtools.data.datautils import make_nbbo_callback
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter
from requests import Session

import pandas as pd

from cryptofeed import FeedHandler
from cryptofeed.callback import TickerCallback, TradeCallback
from cryptofeed.defines import TICKER, TRADES
from cryptofeed.exchanges import Deribit, Bitstamp, Coinbase, Gemini, Kraken
from cryptofeed.feed import Feed

try:
    from .datahandling import ThreadStream
    from .datautils import make_ticker_callback, make_trade_callback
except ImportError:
    from tradingtools.data.datahandling import ThreadStream
    from tradingtools.data.datautils import (
        make_ticker_callback,
        make_trade_callback,
        make_nbbo_callback,
    )


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
        self,
        ticks_consumer: ThreadStream,
        trades_consumer: ThreadStream,
        nbbo_consumer: ThreadStream,
    ) -> None:

        self.ticker_callback = make_ticker_callback(ticks_consumer.add_to_q)
        self.trades_callback = make_trade_callback(trades_consumer.add_to_q)
        self.nbbo_callback = make_nbbo_callback(nbbo_consumer.add_to_q)
        self._consumers_running = True

    def add_instruments(self):
        raise NotImplementedError

    def add_nbbo(self, exchanges: list, instruments: list) -> None:

        self.fh.add_nbbo(exchanges, instruments, self.nbbo_callback)

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

    def add_nbbo(
        self,
        exchanges: List = [Bitstamp, Coinbase, Gemini, Kraken],
        instruments: List = ["BTC-USD", "ETH-USD"],
    ) -> None:
        super().add_nbbo(exchanges, instruments)

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


class HTTPFeed:
    url: str
    session: Session = Session()

    def __init__(self, url, total_retries: int = 5, backoff_factor: int = 2) -> None:

        self.url = url
        retries = Retry(total=total_retries, backoff_factor=backoff_factor)
        self.session.mount("http://", HTTPAdapter(max_retries=retries))
        self.session.mount("https://", HTTPAdapter(max_retries=retries))

    def make_request(self):
        response = self.session.get(self.url)
        return response.json()


if __name__ == "__main__":

    from time import sleep

    api = HTTPFeed(url="https://api.senticrypt.com/v1/bitcoin.json")
    # x = api.make_request()
    # print(json.dumps(x, indent=4))

    stream = ThreadStream()

    producer = api.make_request
    consumer = lambda x: print(x, flush=True)

    stream.add_producer(producer, interval_time=0.1)
    stream.add_consumer(consumer, interval_time=0.5, batched=True)

    # TODO: batch not yet working!!!

    for i in range(100):
        # print(len(stream.get_latest()))
        sleep(0.1)

    exit()

    print("Initializing new OptionsCollectionFeed")
    ocf = OptionsDataFeed(exchange=Deribit)

    ticks_handler = ThreadStream()
    ticks_handler.add_consumer(
        lambda x: print("Ticks: ", len(x), flush=True), interval_time=1, batched=True
    )

    trades_handler = ThreadStream()
    trades_handler.add_consumer(
        lambda x: print("Trades: ", len(x), flush=True), interval_time=1, batched=True
    )

    ocf.add_consumers(ticks_consumer=ticks_handler, trades_consumer=trades_handler)

    print("Syncing instruments")
    ocf.sync_instruments()

    print(f"Running....")
    ocf.run()
