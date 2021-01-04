from uuid import uuid4
import pandas as pd
import ccxt

try:
    from .utils import extract_prices, timestamp_to_string
except:
    from utils import extract_prices, timestamp_to_string


class Broker:
    def __init__(
        self,
        backtest: bool = True,
        exchange_name: str = "binance",
        api_key: str = None,
        secret_key: str = None,
        verbose: bool = True
    ) -> None:
        super().__init__()
        self.exchange = None
        self.backtest = backtest
        self.exchange_name = exchange_name
        self._verbose = verbose

        if not self.backtest:

            confirmation = input(
                "[Broker] backtest = False, are you sure you want to perform actual trades? (type y to continue)"
            )

            if confirmation != "y":
                raise Exception("Please re-run broker init with backtest=False")

        self.exchange = ccxt.binance(
            {
                "apiKey": api_key,
                "secret": secret_key,
                "timeout": 30000,
                "enableRateLimit": True,
            }
        )

        if self._verbose:
            print(
                f"[Broker] logged into Binance -- Status: {self.exchange.fetch_status()['status']}"
            )

    def place_order(self, order: dict, tick: list = []) -> dict:

        # orders = {"symbol": "BTCUSD", "order_type": "buy", "volume": 10}

        adjusted_order = self._adjust_order_for_exchange(order)

        if self.backtest:
            settlement = self._simulate_settlement(order, tick)
        else:

            order_response = self._place_market_order(
                symbol=adjusted_order["symbol"],
                side=adjusted_order["side"],
                amount=adjusted_order["amount"],
            )

            settlement = {
                "order_id": order["order_id"],
                "exchange_order_id": order_response["id"],
                "timestamp": order_response["datetime"],
                "price": order_response["price"],
                "amount": order_response["amount"],
                "cost": order_response["cost"],
                "average_cost": order_response["average"],
                "fee": order_response["fee"]["cost"],
                "fee_currency": order_response["fee"]["currency"],
            }

        return settlement

    def _adjust_order_for_exchange(self, order: dict) -> dict:

        adjusted_order = order.copy()

        if self.exchange_name == "binance":

            symbol_mapping = {'BTCEUR': 'BTC/EUR', 'BTCUSD': 'BTC/USDT'}

            if order['symbol'] in symbol_mapping:
                adjusted_order['symbol'] = symbol_mapping[order['symbol']]

        return adjusted_order

    def _place_market_order(self, symbol, side, amount):

        # extra params and overrides if needed
        params = {
            "test": self.backtest,  # test if it's valid, but don't actually place it
        }

        order_response = self.exchange.create_order(
            symbol=symbol,
            type="market",
            side=side,
            amount=amount,
            price=None,
            params=params,
        )

        if self._verbose:
            print(f"[Broker] order response: {order_response}")

        return order_response

    def _simulate_settlement(self, order: dict, tick: list = []) -> dict:

        prices_close = extract_prices(tick, "close")
        prices_high = extract_prices(tick, "high")

        if self.exchange_name == "binance":
            fee = prices_close[order["symbol"]] / 1000
        else:
            fee = prices_close[order["symbol"]] / 100

        price = prices_high[order["symbol"]]

        settlement = {
            "order_id": order["order_id"],
            "symbol": order["symbol"],
            "exchange_order_id": uuid4().hex,
            "timestamp": timestamp_to_string(pd.Timestamp.now()),
            "fills": [],
            "price": price,
            "amount": order["amount"],
            "filled": order["amount"],
            "cost": price * order["amount"] + fee,
            "average_cost": price * order["amount"],
            "fee": fee,
            "fee_currency": "EUR",
        }

        return settlement


if __name__ == "__main__":

    order = {
        "order_id": "448c10fd7af44d49b5667b90e412e1f6",
        "symbol": "BTC/USDT",
        "order_type": "sell",
        "volume": 1.1874331869272101,
    }

    tick = [
        {
            "symbol": "BTC/USDT",
            "open": 3902.52,
            "high": 3908.0,
            "low": 3902.25,
            "close": 3902.25,
            "volume": 0.25119066,
            "timestamp": pd.Timestamp.now(),
        },
        {
            "symbol": "ETH/USD",
            "open": 3902.52,
            "high": 3908.0,
            "low": 3902.25,
            "close": 3902.25,
            "volume": 0.25119066,
            "timestamp": pd.Timestamp.now(),
        },
    ]

    import json

    with open("./secrets.json", "r") as in_file:
        secrets = json.load(in_file)["binance"]

    brkr = Broker(
        backtest=True,
        exchange_name="binance",
        api_key=secrets["api_key"],
        secret_key=secrets["secret_key"],
    )
    sttl = brkr.place_order(order, tick)
    print(sttl)

    print('\n\n----')
    orders = brkr.exchange.fetch_orders("BTC/EUR")
    print(orders)
    print(len(orders))
