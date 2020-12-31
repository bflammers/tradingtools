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

    def _validate_inputs(self) -> None:

        # Check if inputs are consistent with one another
        pass

    def initialize_ticker(self) -> None:
        self.i = 0
        self.ticker = self.dataloader.get_ticker()

    def run(self) -> None:

        if self.ticker is None:
            raise Exception("First initialize ticker")

        for tick in self.ticker:

            # Pass to strategy -> optimal_positions
            optimal_positions = self.strategy.execute_on_tick(tick)
        
            # Pass to porfolio -> order
            orders = self.portfolio.update(tick, optimal_positions)

            for order in orders:

                # ASYNC! -> callback to add_settlement

                # Pass to broker -> settlement back to portfolio
                settlement = self.broker.place_order(order, tick)

                self.portfolio.add_settlement(
                    order_id=settlement["order_id"],
                    price=settlement["price"],
                    fee=settlement["fee"],
                    timestamp_settlement=settlement["timestamp"],
                )

            # Increment counter
            self.i += 1

            if self.verbose:
                if self.i % 1000 == 0:
                    print(self.portfolio)
                    print(self.dataloader)


if __name__ == "__main__":

    import signal

    def quit_gracefully(*args):
        print('quitting loop')
        exit(0)

    signal.signal(signal.SIGINT, quit_gracefully)

    try:
        from .data import HistoricalOHLCLoader
        from .strategy import MovingAverageCrossOverSingle
    except ImportError:
        from data import HistoricalOHLCLoader
        from strategy import MovingAverageCrossOverSingle

    try: 

        p = "./data/cryptodatadownload/gemini/price"
        dl = HistoricalOHLCLoader("BTCUSD", p, extra_pattern="2020")

        strat = MovingAverageCrossOverSingle(symbol="BTCUSD")
        pf = Portfolio(5000)
        brkr = Broker()

        pipeline = Pipeline(dataloader=dl, strategy=strat, portfolio=pf, broker=brkr)

        pipeline.initialize_ticker()

        pipeline.run()

    except KeyboardInterrupt:
        quit_gracefully()
