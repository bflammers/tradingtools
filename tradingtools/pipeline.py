import pandas as pd
import numpy as np

import json
from tradingtools.portfolio import Symbol

from tradingtools.strategy import MovingAverageCrossOverOHLC

try:
    from .data import HistoricalOHLCLoader
    from .strategy import Strategy
    from .portfolio import Portfolio
    from .broker import Broker
except:
    from data import HistoricalOHLCLoader
    from strategy import Strategy
    from portfolio import Portfolio
    from broker import Broker


class Pipeline:
    def __init__(
        self,
        dataloader: HistoricalOHLCLoader,
        strategy: Strategy,
        portfolio: Portfolio,
        broker: Broker,
        risk_management=None,
        sync_exchange: bool = True,
        verbose=True,
    ) -> None:
        super().__init__()

        self.dataloader = dataloader
        self.strategy = strategy
        self.risk_management = risk_management
        self.portfolio = portfolio
        self.broker = broker

        self.verbose = verbose

        self._validate_inputs()

        self.i = 0
        self.ticker = None
        self._initialize_ticker()

        # Sync with exchange
        if sync_exchange:
            self.sync_exchange(initialize=True)

    def _validate_inputs(self) -> None:

        # Check if inputs are consistent with one another
        pass

    def _initialize_ticker(self) -> None:
        self.i = 0
        self.ticker = self.dataloader.get_ticker()

    def run(self) -> None:

        if self.ticker is None:
            raise Exception("First initialize ticker")

        for tick in self.ticker:
            self.single_run(tick)

    def sync_exchange(self, initialize: bool = False) -> None:

        # Sync with exchange
        symbol_amounts = self.broker.get_symbol_amounts()
        tick = self.dataloader.get_single_tick()
        self.portfolio.initialize(symbol_amounts, tick)

    def single_run(self, tick) -> None:

        # Pass to strategy -> optimal_positions
        optimal_positions = self.strategy.execute_on_tick(tick)

        # Pass to porfolio -> order
        orders = self.portfolio.update(tick, optimal_positions)

        for order in orders:

            # TODO: make ASYNC w/ callback to add_settlement

            # Pass to broker -> settlement back to portfolio
            settlement = self.broker.place_order(order, tick)

            self.portfolio.settle_order(
                trading_pair=settlement["trading_pair"],
                order_id=settlement["order_id"],
                order_value=settlement["cost"],
                price_settlement=settlement["price"],
                timestamp_settlement=settlement["timestamp"],
                fee=settlement["fee"],
                fee_currency=settlement["fee_currency"],
            )

        # Increment counter
        self.i += 1

        if self.verbose:
            print(self.portfolio)
            print(self.dataloader)


if __name__ == "__main__":

    import signal

    def quit_gracefully(*args):
        print("quitting loop")
        exit(0)

    signal.signal(signal.SIGINT, quit_gracefully)

    try:
        from .data import HistoricalOHLCLoader
        from .strategy import MovingAverageCrossOverClose
    except ImportError:
        from data import HistoricalOHLCLoader
        from strategy import MovingAverageCrossOverClose

    try:

        backtest = True

        p = "./data/cryptodatadownload/binance/price"
        dl = HistoricalOHLCLoader("BTCUSD", p, extra_pattern="dev")
        dl.df["trading_pair"] = "BTC/EUR"

        strat = MovingAverageCrossOverOHLC(
            trading_pair="BTC/EUR", trading_amount=0.1, metrics=["low", "close"]
        )
        pf = Portfolio(
            5000, results_parent_dir="./runs/" + "backtest" if backtest else ""
        )

        with open("./secrets.json", "r") as in_file:
            secrets = json.load(in_file)["binance"]

        brkr = Broker(
            backtest=backtest,
            exchange_name="binance",
            api_key=secrets["api_key"],
            secret_key=secrets["secret_key"],
        )

        pipeline = Pipeline(dataloader=dl, strategy=strat, portfolio=pf, broker=brkr)

        pipeline.run()

    except KeyboardInterrupt:
        quit_gracefully()
