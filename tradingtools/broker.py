import pandas as pd

try:
    from .utils import extract_prices
except:
    from utils import extract_prices

class Broker:
    def __init__(self, backtest: bool = True) -> None:
        super().__init__()

        if not backtest:
            confirmation = input(
                "[Broker] backtest = False, are you sure you want to perform actual trades? (type y to continue)"
            )
            if confirmation != "y":
                raise Exception("[Broker] please re-run with backtest = True")

        self.backtest = backtest

    def place_order(self, order: dict, tick: list = []) -> dict:

        # orders = {"symbol": "BTCUSD", "order_type": "buy", "volume": 10}

        if self.backtest:

            prices_close = extract_prices(tick, "close")
            prices_high = extract_prices(tick, "high")

            fee = prices_close[order["symbol"]] / 100
            price = prices_high[order["symbol"]]

            return {
                "order_id": order["order_id"],
                "timestamp": pd.Timestamp.now(),
                "price": price,
                "fee": fee
            }

        raise NotImplementedError("[Broker] not implemented without backtest")


if __name__ == "__main__":

    order = {
        "order_id": "448c10fd7af44d49b5667b90e412e1f6",
        "symbol": "BTCUSD",
        "order_type": "sell",
        "volume": 1.1874331869272101,
    }

    tick = [
        {
            "symbol": "BTCUSD",
            "open": 3902.52,
            "high": 3908.0,
            "low": 3902.25,
            "close": 3902.25,
            "volume": 0.25119066,
            "timestamp": pd.Timestamp.now(),
        },
        {
            "symbol": "ETHUSD",
            "open": 3902.52,
            "high": 3908.0,
            "low": 3902.25,
            "close": 3902.25,
            "volume": 0.25119066,
            "timestamp": pd.Timestamp.now(),
        },
    ]

    brkr = Broker()
    sttl = brkr.place_order(order, tick)
