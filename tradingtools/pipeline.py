import pandas as pd
import numpy as np

from uuid import uuid4
from pathlib import Path

import threading
import json

try:
    from .data import HistoricalOHLCLoader
    from .strategy import Strategy, MovingAverageCrossOverOHLC
    from .portfolio import Portfolio
    from .broker import Broker
    from .utils import CSVWriter
except:
    from data import HistoricalOHLCLoader
    from strategy import Strategy, MovingAverageCrossOverOHLC
    from portfolio import Portfolio
    from broker import Broker
    from utils import CSVWriter


class Pipeline:
    def __init__(
        self,
        dataloader: HistoricalOHLCLoader,
        strategy: Strategy,
        portfolio: Portfolio,
        broker: Broker,
        risk_management=None,
        sync_exchange: bool = True,
        results_parent_dir: str = "./runs",
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
        self.ticker = None
        self._initialize_ticker()

        # Sync with exchange
        if sync_exchange:
            self._sync_exchange(initialize=True)

        # Create directory for results
        now = pd.Timestamp.now().strftime("%Y%m%d_%H%M")
        ts_uuid = f"{now}_{uuid4().hex}"
        self._results_dir = (Path(results_parent_dir) / ts_uuid).absolute()
        self._results_dir.mkdir(parents=True, exist_ok=True)

        if self.verbose:
            print(f"[Portfolio] results directory created: {self._results_dir}")

        self._tick_writer = CSVWriter(
            path=self._results_dir / f"{now}_ticks.csv",
            columns=[
                "id",
                "trading_pair",
                "timestamp",
                "open",
                "high",
                "low",
                "close",
                "volume",
            ],
        )

        self._opt_positions_writer = CSVWriter(
            path=self._results_dir / f"{now}_optimal_positions.csv",
            columns=["id", "timestamp", "symbol", "amount"],
        )
        self._prev_optimal_positions = None

        self._orders_writer = CSVWriter(
            path=self._results_dir / f"{now}_orders.csv",
            columns=[
                "order_id",
                "trading_pair",
                "side",
                "amount",
                "timestamp_tick",
                "price_execution",
                "cost_execution",
                "timestamp_execution",
            ],
        )

        self._settlements_writer = CSVWriter(
            path=self._results_dir / f"{now}_settlements.csv",
            columns=[
                "order_id",
                "trading_pair",
                "status",
                "side",
                "amount",
                "timestamp_tick",
                "price_execution",
                "cost_execution",
                "timestamp_execution",
                "price_settlement",
                "timestamp_settlement",
                "fee",
                "fee_currency",
                "fee_reference_currency",
                "slippage",
                "order_value",
            ],
        )

    def _validate_inputs(self) -> None:

        # Check if inputs are consistent with one another
        pass

    def _initialize_ticker(self) -> None:
        self.ticker = self.dataloader.get_ticker()

    def sync_exchange(self, threaded: bool = False) -> None:

        if threaded:
            job_thread = threading.Thread(target=self._sync_exchange)
            job_thread.start()
        else:
            self._sync_exchange()

    def _sync_exchange(self, initialize: bool = False) -> None:

        # Sync with exchange
        symbol_amounts = self.broker.get_symbol_amounts()
        tick = self.dataloader.get_single_tick()
        self.portfolio.initialize(symbol_amounts, tick)

    def run(self) -> None:

        if self.ticker is None:
            raise Exception("First initialize ticker")

        for tick in self.ticker:
            self._execute(tick)

    def single_run(self, threaded: bool = False) -> None:

        if threaded:
            job_thread = threading.Thread(target=self._single_run)
            job_thread.start()
        else:
            self._single_run()

    def _single_run(self) -> None:

        if self.ticker is None:
            raise Exception("First initialize ticker")

        # Get next tick
        tick = self.ticker.__next__()

        # Execute pipeline
        self._execute(tick)

    def _execute(self, tick: list) -> None:

        # Write tick to file
        self._tick_writer.append_multiple(tick, add_uuid=True)

        # Pass to strategy -> optimal_positions
        optimal_positions = self.strategy.execute_on_tick(tick)

        # Write optimal postions if changed
        if self._prev_optimal_positions != optimal_positions:
            opt_positions_list = [
                {"symbol": s, "amount": a} for s, a in optimal_positions.items()
            ]
            self._opt_positions_writer.append_multiple(
                opt_positions_list, add_timestamp=True, add_uuid=True
            )
            self._prev_optimal_positions = optimal_positions

        # Pass to porfolio -> order
        orders = self.portfolio.update(tick, optimal_positions)

        # Write order to log if any
        if orders:
            self._orders_writer.append_multiple(orders)

        for order in orders:

            # TODO: make ASYNC w/ callback to add_settlement

            # Pass to broker -> settlement back to portfolio
            settlement = self.broker.place_order(order, tick)

            settled_order = self.portfolio.settle_order(
                trading_pair=settlement["trading_pair"],
                order_id=settlement["order_id"],
                status=settlement["status"],
                order_value=settlement["cost"],
                price_settlement=settlement["price"],
                timestamp_settlement=settlement["timestamp"],
                fee=settlement["fee"],
                fee_currency=settlement["fee_currency"],
            )

            # Write to csv
            self._settlements_writer.append(settled_order)

        if self.verbose:
            print(self.portfolio)
            print(self.dataloader)

    def get_ticks(self) -> pd.DataFrame:
        df = self._tick_writer.read()
        return df

    def get_optimal_positions(self) -> pd.DataFrame:
        df = self._opt_positions_writer.read()
        return df

    def get_orders(self) -> pd.DataFrame:
        df = self._orders_writer.read()
        return df

    def get_settled_orders(self) -> pd.DataFrame:
        df = self._settlements_writer.read()
        return df


if __name__ == "__main__":

    import signal

    def quit_gracefully(*args):
        print("quitting loop")
        exit(0)

    signal.signal(signal.SIGINT, quit_gracefully)

    try:
        from .data import HistoricalOHLCLoader
        from .strategy import MovingAverageCrossOverOHLC
    except ImportError:
        from data import HistoricalOHLCLoader
        from strategy import MovingAverageCrossOverOHLC

    try:

        backtest = True

        p = "./data/cryptodatadownload/gemini/price"
        dl = HistoricalOHLCLoader("BTCUSD", p, extra_pattern="2019")
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
