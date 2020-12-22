import pandas as pd
import numpy as np

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
    ) -> None:
        super().__init__()

        self.dataloader = dataloader
        self.strategy = strategy
        self.risk_management = risk_management
        self.portfolio = portfolio
        self.broker = broker

        self._validate_inputs()

        self.i = 0
        self.ticker = None

    def _validate_inputs(self) -> None:

        # Check if inputs are consistent with one another
        pass

    def initialize_ticker(self) -> None:
        self.i = 0
        self.ticker = self.dataloader.get_ticker()

    def run(self, backtest=True) -> None:

        if self.ticker is None:
            raise Exception("First initialize ticker")

        for tick in self.ticker:
            self.single_run(tick, backtest)

    def single_run(self, tick: dict, backtest: bool = False) -> None:

        # Pass to strategy -> optimal_position
        optimal_position = self.strategy.execute_on_tick(tick)
        order = [("BTCUSD", 10), ("ETHUSD", 0)]

        # Pass to risk management -> optimal_position
        pass

        # Pass to porfolio -> order
        order = self.portfolio.update(optimal_position)
        order = [("BTCUSD", "buy", 10), ("ETHUSD", "sell", 5)]

        # Pass to broker -> settlement back to portfolio
        settlement = self.broker.place_order(order, backtest)
        settlement = [
            ("order_id", "ex_timestamp", "ex_price"),
            ("order_id", "ex_timestamp", "ex_price"),
        ]
        self.portfolio.add_settlement(settlement)

        # Increment counter
        self.i += 1


if __name__ == "__main__":

    try:
        from .data import HistoricalOHLCLoader
        from .strategy import MovingAverageCrossOverSingle
    except ModuleNotFoundError:
        from data import HistoricalOHLCLoader
        from strategy import MovingAverageCrossOverSingle

    p = "./data/cryptodatadownload/gemini/price"
    dl = HistoricalOHLCLoader("BTCUSD", p, extra_pattern="2019")

    strat = MovingAverageCrossOverSingle(symbol="BTCUSD")
    pf = Portfolio()
    brkr = Broker()

    pipeline = Pipeline(dataloader=dl, strategy=strat, portfolio=pf, broker=brkr)

    pipeline.initialize_ticker()
    pipeline.run()
