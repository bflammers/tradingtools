import numpy as np
from collections import deque
from decimal import Decimal


class Strategy:
    def __init__(self) -> None:
        super().__init__()
        self.i = 0

    def execute_on_tick(self, tick: list, context: dict = None) -> dict:
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


class MovingAverageCrossOverClose(Strategy):
    def __init__(
        self,
        trading_pair: str,
        trading_amount: Decimal,
        n_short: int = 100,
        n_long: int = 300,
    ) -> None:
        super().__init__()
        self.trading_pair = trading_pair
        self.n_short = n_short
        self.n_long = n_long
        self.trading_amount = Decimal(trading_amount)

        # Initialize arrays
        self.short_que = deque(self.n_short * [Decimal("0")], maxlen=self.n_short)
        self.long_que = deque(self.n_long * [Decimal("0")], maxlen=self.n_long)

    def execute_on_tick(self, tick: list) -> dict:

        # Hacky - but works for now
        tick = [x for x in tick if x["trading_pair"] == self.trading_pair][0]

        # Skip first 300 days to get full windows
        self.i += 1

        opt_position = self._execute(tick["close"])

        return opt_position

    def _execute(self, x):

        # Update lists with latest Closing price
        self.long_que.append(x)
        self.short_que.append(x)

        # Calculate means
        short_mavg = np.mean(self.short_que)
        long_mavg = np.mean(self.long_que)

        # Trading logic
        if self.i > self.n_long:
            if short_mavg >= long_mavg:
                return {self.trading_pair: self.trading_amount}

        return {self.trading_pair: Decimal(0)}


class MovingAverageCrossOverOHLC(Strategy):
    def __init__(
        self,
        trading_pair: str,
        trading_amount: Decimal,
        n_short: int = 100,
        n_long: int = 300,
        metrics: list = ["open", "high", "low", "close"]
    ) -> None:
        super().__init__()
        self.trading_pair = trading_pair
        self.n_short = n_short
        self.n_long = n_long
        self.trading_amount = Decimal(trading_amount)

        self.metrics = metrics

        assert set(metrics).issubset({"open", "high", "low", "close"})

        # Initialize arrays
        for metric in self.metrics:
            setattr(
                self,
                f"short_que_{metric}",
                deque(self.n_short * [Decimal("0")], maxlen=self.n_short),
            )
            setattr(
                self,
                f"long_que_{metric}",
                deque(self.n_long * [Decimal("0")], maxlen=self.n_long),
            )

    def execute_on_tick(self, tick: list) -> dict:

        # Hacky - but works for now
        tick = [x for x in tick if x["trading_pair"] == self.trading_pair][0]

        # Skip first n_long days to get full windows
        self.i += 1

        return self._execute(tick)

    def _execute(self, tick):

        n_active = 0

        for metric in self.metrics:
            
            short_que = getattr(self, f"short_que_{metric}")
            long_que = getattr(self, f"long_que_{metric}")

            # Update lists with latest metric
            short_que.append(tick[metric])
            long_que.append(tick[metric])

            # Calculate means
            short_mavg = np.mean(short_que)
            long_mavg = np.mean(long_que)

            # Trading logic
            if short_mavg >= long_mavg:
                n_active += 1

        if self.i > self.n_long and n_active == len(self.metrics):
            return {self.trading_pair: self.trading_amount}

        return {self.trading_pair: Decimal(0)}


if __name__ == "__main__":

    strat = MovingAverageCrossOverOHLC("BTCUSD", 0.1)

    tick = [
        {
            "symbol": "BTCUSD",
            "open": Decimal(3902.52),
            "high": Decimal(3908.0),
            "low": Decimal(3902.25),
            "close": Decimal(3902.25),
            "volume": Decimal(0.25119066),
            "timestamp": "2019-01-02 23:25:00",
        },
        {
            "symbol": "ETHUSD",
            "open": Decimal(3902.52),
            "high": Decimal(3908.0),
            "low": Decimal(3902.25),
            "close": Decimal(3902.25),
            "volume": Decimal(0.25119066),
            "timestamp": "2019-01-02 23:25:00",
        },
    ]

    for i in range(301):
        optimal_positions = strat.execute_on_tick(tick)

        if optimal_positions["BTCUSD"] > 0:
            print(f"{i} - {optimal_positions}")
