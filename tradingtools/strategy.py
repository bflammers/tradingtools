import pandas as pd
import numpy as np


class Strategy:
    def __init__(self) -> None:
        self.i = 0

    def execute_on_tick(self, tick: dict, context: dict = None) -> list:
        """Executes the strategy on each tick of data (ohlc + extra context)
        manages its own state and produces an optimal portfolio allocation

        Args:
            tick (dict): ohlc data
            context (dict): contextual data

        Raises:
            NotImplementedError: Base class, use specific strategy implementation
        """
        raise NotImplementedError


class MovingAverageCrossOverSingle(Strategy):
    def __init__(self, symbol: str, n_short: int = 300, n_long: int = 900) -> None:
        super().__init__()
        self.symbol = symbol
        self.n_short = n_short
        self.n_long = n_long

        # Initialize arrays
        self.short_list = np.zeros(self.n_short)
        self.long_list = np.zeros(self.n_long)
        self.open_pos = False

    def execute_on_tick(self, tick) -> list:

        # Skip first 300 days to get full windows
        self.i += 1

        # Update lists with latest Closing price
        self.short_list = np.append(self.short_list[1:], tick["close"])
        self.long_list = np.append(self.long_list[1:], tick["close"])

        # Calculate means
        short_mavg = self.short_list.mean()
        long_mavg = self.long_list.mean()

        # Trading logic
        if self.i > 300:
            if short_mavg > long_mavg and not self.open_pos:
                self.open_pos = True
                return [(self.symbol, 10)]
            elif short_mavg < long_mavg and self.open_pos:
                self.open_pos = False
                return [(self.symbol, 0)]

        return []


if __name__ == "__main__":

    strat = MovingAverageCrossOverSingle("BTCUSD")

    for i in range(310):
        strat.execute_on_tick(pd.Series({"close": 100}))
