from uuid import uuid4
import pandas as pd
import ccxt
from decimal import Decimal

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
        verbose: bool = True,
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
                "options": {
                    "adjustForTimeDifference": True,  # resolves the recvWindow timestamp error
                    "recvWindow": 59999,  # resolves the recvWindow timestamp error
                },
            }
        )

        if self._verbose:
            print(
                f"[Broker] logged into Binance -- Status: {self.exchange.fetch_status()['status']}"
            )

    def place_order(self, order: dict, tick: list = []) -> dict:

        # orders = {"trading_pair": "BTCUSD", "side": "buy", "amount": 10}

        adjusted_order = self._adjust_order_for_exchange(order)

        if self.backtest:
            settlement = self._simulate_settlement(order, tick)
        else:

            order_response = self._place_market_order(
                trading_pair=adjusted_order["trading_pair"],
                side=adjusted_order["side"],
                amount=adjusted_order["amount"],
            )

            settlement = {
                "order_id": order["order_id"],
                "exchange_order_id": order_response["id"],
                "trading_pair": order_response["trading_pair"],
                "timestamp": order_response["datetime"],
                "price": Decimal(order_response["price"]),
                "amount": Decimal(order_response["amount"]),
                "cost": Decimal(order_response["cost"]),
                "average_cost": Decimal(order_response["average"]),
                "fee": Decimal(order_response["fee"]["cost"]),
                "fee_currency": order_response["fee"]["currency"],
            }

        return settlement

    def get_symbol_amounts(self):
        balance = self.exchange.fetch_balance()
        symbol_amounts = {k: Decimal(v) for k, v in balance["free"].items() if v > 0}
        return symbol_amounts

    def _adjust_order_for_exchange(self, order: dict) -> dict:

        adjusted_order = order.copy()

        if self.exchange_name == "binance":

            trading_pair_mapping = {"BTCEUR": "BTC/EUR", "BTCUSD": "BTC/USDT"}

            if order["trading_pair"] in trading_pair_mapping:
                adjusted_order["trading_pair"] = trading_pair_mapping[
                    order["trading_pair"]
                ]

        return adjusted_order

    def _place_market_order(self, trading_pair, side, amount):

        # extra params and overrides if needed
        params = {
            "test": self.backtest,  # test if it's valid, but don't actually place it
        }

        order_response = self.exchange.create_order(
            trading_pair=trading_pair,
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

        prices_high = extract_prices(tick, "high")

        if self.exchange_name == "binance":
            fee = order["cost_execution"] / 1000
        else:
            fee = order["cost_execution"] / 100

        price = prices_high[order["trading_pair"]]

        settlement = {
            "order_id": order["order_id"],
            "trading_pair": order["trading_pair"],
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
        "order_id": uuid4().hex,
        "trading_pair": "BTC/EUR",
        "side": "buy",
        "amount": Decimal("0.01"),
        "timestamp_tick": timestamp_to_string(pd.Timestamp.now()),
        "price_execution": Decimal("32009"),
        "cost_execution": Decimal("0.01") * Decimal("32009"),
        "timestamp_execution": timestamp_to_string(pd.Timestamp.now()),
    }

    tick = [
        {
            "trading_pair": "BTC/EUR",
            "open": Decimal(3902.52),
            "high": Decimal(3908.0),
            "low": Decimal(3902.25),
            "close": Decimal(3902.25),
            "volume": Decimal(0.25119066),
            "timestamp": timestamp_to_string(pd.Timestamp.now()),
        },
        {
            "trading_pair": "ETH/USD",
            "open": Decimal(3902.52),
            "high": Decimal(3908.0),
            "low": Decimal(3902.25),
            "close": Decimal(3902.25),
            "volume": Decimal(0.25119066),
            "timestamp": timestamp_to_string(pd.Timestamp.now()),
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

    print("\n\n----")
    orders = brkr.exchange.fetch_orders("BTC/EUR")
    print(orders)
    print(len(orders))
