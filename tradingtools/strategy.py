
import pandas as pd
import numpy as np
from collections import deque

class Strategy:
    def __init__(self) -> None:
        self.i = 0

    def execute_on_tick(self, tick: list, context: dict = None) -> list:
        """Executes the strategy on each tick of data (ohlc + extra context)
        manages its own state and produces a list of optimal portfolio allocations
        with one element per symbol

        Args:
            tick (list): list of ohlc data
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
        self.short_que = deque(self.n_short* [0.0], maxlen=self.n_short)
        self.long_que = deque(self.n_long* [0.0], maxlen=self.n_long)

        self.x = 0

    def execute_on_tick(self, tick: list) -> list:

        # Hacky - but works for now
        tick = [x for x in tick if x["symbol"] == self.symbol][0]

        # Skip first 300 days to get full windows
        self.i += 1

        return self._execute(tick['close'])

    def _execute(self, x):

        # Moving average
        self.x = self.x * (99/100) + x * (1/100)

        # Update lists with latest Closing price
        self.long_que.append(self.x)
        self.short_que.append(self.x)

        # Calculate means
        short_mavg = np.mean(self.short_que)
        long_mavg = np.mean(self.long_que)

        # Trading logic
        if self.i > self.n_long:
            if short_mavg > long_mavg:
                return [{"symbol": self.symbol, "volume": 0.1}]        

        return [{"symbol": self.symbol, "volume": 0}]


if __name__ == "__main__":

    strat = MovingAverageCrossOverSingle("BTCUSD")

    tick = [
        {
            "symbol": "BTCUSD",
            "open": 3902.52,
            "high": 3908.0,
            "low": 3902.25,
            "close": 3902.25,
            "volume": 0.25119066,
            "timestamp": "2019-01-02 23:25:00",
        },
        {
            "symbol": "ETHUSD",
            "open": 3902.52,
            "high": 3908.0,
            "low": 3902.25,
            "close": 3902.25,
            "volume": 0.25119066,
            "timestamp": "2019-01-02 23:25:00",
        },
    ]

    for i in range(301):
        optimal_positions = strat.execute_on_tick(tick)

        if optimal_positions[0]['volume'] > 0:
            print(f"{i} - {optimal_positions}")
