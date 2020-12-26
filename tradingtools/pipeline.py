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

    def single_run(self, tick: list, backtest: bool = False) -> None:

        # [{'symbol': 'BTCUSD', 'open': 3902.52, 'high': 3908.0, 'low': 3902.25, 'close': 3902.25, 'volume': 0.25119066, 'timestamp': Timestamp('2019-01-02 23:25:00')}

        # Pass to strategy -> optimal_positions
        optimal_positions = self.strategy.execute_on_tick(tick)
        optimal_positions = [
            {"symbol": "BTCUSD", "volume": 10},
            {"symbol": "ETHUSD", "volume": 0},
        ]
        prices = self._extract_prices(tick)

        # Pass to risk management -> optimal_position
        pass

        # Pass to porfolio -> order
        orders = self.portfolio.update(prices, optimal_positions)
        orders = [
            {"symbol": "BTCUSD", "order_type": "buy", "volume": 10},
            {"symbol": "ETHUSD", "order_type": "sell", "volume": 5},
        ]
        print(orders)
        # Pass to broker -> settlement back to portfolio
        settlement = self.broker.place_order(orders, backtest)
        settlement = [
            {"order_id": "2jdiejd", "timestamp": "2020-10-10 11:10", "price": 189.90},
            {"order_id": "0i3ek3m", "timestamp": "2020-10-10 11:10", "price": 129.00},
        ]
        self.portfolio.add_settlement()

        # Increment counter
        self.i += 1

    @staticmethod
    def _extract_prices(tick):
        return {t['symbol']: t['close'] for t in tick}


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
