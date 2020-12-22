import pandas as pd
import numpy as np

try:
    from .data import HistoricalOHLCLoader
    from .strategy import MovingAverageCrossOver
    from .portfolio import Portfolio
except:
    from data import HistoricalOHLCLoader
    from strategy import MovingAverageCrossOver
    from portfolio import Portfolio


class Backtest:
    def __init__(
        self, data_path="./data/cryptodatadownload/gemini/price", extra_pattern="2019"
    ):

        self.dl = HistoricalOHLCLoader(
            symbol="BTCUSD",
            path=data_path,
            extra_pattern=extra_pattern,
        )
        self.strategy = MovingAverageCrossOver()
        self.ticker = self.dl.get_ticker()
        self.pf = Portfolio()

    def run(self):
        for row in self.ticker:
            order = self.strategy.execute_on_tick(row)

            if order != 0:
                print(f"[Backtest] >> order: {order}")
                self.pf.add_order(
                    symbol="BTCUSD",
                    timestamp_execution=row.name,
                    volume=order,
                    price_execution=row["Close"],
                    price_settlement=row["Close"],
                    timestamp_settlement=row.name
                )

            print(self.pf)


if __name__ == "__main__":

    bt = Backtest()
    bt.run()
