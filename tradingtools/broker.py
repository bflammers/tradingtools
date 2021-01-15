from uuid import uuid4
import pandas as pd
import ccxt
from decimal import Decimal
from dataclasses import dataclass

try:
    from .utils import extract_prices, timestamp_to_string, warnings
except:
    from utils import extract_prices, timestamp_to_string, warnings


@dataclass
class Settlement:
    order_id: str
    status: str
    trading_pair: str
    exchange_order_id: str = None
    timestamp: str = None
    value: Decimal = None
    price: Decimal = None
    amount: Decimal = None
    cost: Decimal = None
    average_cost: Decimal = None
    fee: Decimal = None
    fee_currency: str = None


@dataclass
class Order:
    order_id: str
    trading_pair: str
    status: str
    side: str
    amount: Decimal
    timestamp_tick: str
    price_execution: Decimal
    cost_execution: Decimal
    timestamp_execution: Decimal
    settled: bool = False
    value_settlement: Decimal = None
    price_settlement: Decimal = None
    timestamp_settlement: Decimal = None
    cost_settlement: Decimal = None
    average_cost_settlement: Decimal = None
    exchange_order_id: str = None
    fee: Decimal = None
    fee_currency: str = None

    def settle(self, settlement: Settlement) -> None:

        self.value_settlement = settlement.value
        self.price_settlement = settlement.price
        self.timestamp_settlement = settlement.timestamp
        self.cost_settlement = settlement.cost
        self.average_cost_settlement = settlement.average_cost
        self.exchange_order_id = settlement.exchange_order_id
        self.fee = settlement.fee
        self.fee_currency = settlement.fee_currency
        self.settled = True


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
        self._min_order_amount = Decimal(0.001)

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

    def place_order(self, order: Order, tick: list = []) -> Settlement:

        # orders = {"trading_pair": "BTCUSD", "side": "buy", "amount": 10}

        # adjusted_order = self._prep_order_for_exchange(order)

        if self.backtest:
            order_response = self._simulate_order_response(order, tick)
        else:

            try:
                order_response = self._place_market_order(
                    trading_pair=order.trading_pair,
                    side=order.side,
                    amount=order.amount,
                )
            except Exception as e:
                warnings.warn(f'[Broker.place_order] failed with order: \n\t{order} and messsage: \n\t{e}')
                settlement = Settlement(
                    order_id=order.order_id,
                    status="failed",
                    trading_pair=order.trading_pair,
                )
                return settlement

        settlement = Settlement(
            order_id=order.order_id,
            status="pending",
            trading_pair=order.trading_pair,
            exchange_order_id=order_response["id"],
            timestamp=order_response["datetime"],
            price=Decimal(order_response["price"]),
            value = Decimal(order_response["price"]) * Decimal(order_response["amount"]),
            amount=Decimal(order_response["amount"]),
            cost=Decimal(order_response["cost"]),
            average_cost=Decimal(order_response["average"]),
            fee=Decimal(order_response["fee"]["cost"]),
            fee_currency=order_response["fee"]["currency"],
        )

        return settlement

    def get_symbol_amounts(self):
        balance = self.exchange.fetch_balance()
        symbol_amounts = {k: Decimal(v) for k, v in balance["free"].items() if v > 0}
        return symbol_amounts

    # def _prep_order_for_exchange(self, order: Order) -> dict:

    #     adjusted_order = order.copy()

    #     if self.exchange_name == "binance":

    #         if order["amount"] < (self._min_order_amount - Decimal(0.00001)):
    #             warnings.warn(
    #                 f"[Broker._prep_order_for_exchange] order {order['order_id']} amount smaller than exchange smallest order amount"
    #             )

    #         trading_pair_mapping = {"BTCEUR": "BTC/EUR", "BTCUSD": "BTC/USDT"}

    #         if order["trading_pair"] in trading_pair_mapping:
    #             adjusted_order["trading_pair"] = trading_pair_mapping[
    #                 order["trading_pair"]
    #             ]

    #     return adjusted_order

    def _place_market_order(
        self, trading_pair: str, side: str, amount: Decimal
    ) -> dict:

        # extra params and overrides if needed
        params = {
            "test": self.backtest,  # test if it's valid, but don't actually place it
        }

        # Create market order through ccxt with specified exchange
        order_response = self.exchange.create_order(
            symbol=trading_pair,
            type="market",
            side=side,
            amount=amount,
            price=None,
            params=params,
        )

        if self._verbose:
            print(f"[Broker] order response: {order_response}")

        return order_response

    def _simulate_order_response(self, order: Order, tick: list = []) -> dict:

        prices_high = extract_prices(tick, "high")

        if self.exchange_name == "binance":
            fee = order.cost_execution / 1000
        else:
            fee = order.cost_execution / 100

        price = prices_high[order.trading_pair]

        order_response = {
            "id": uuid4().hex,
            "datetime": timestamp_to_string(pd.Timestamp.now()),
            "price": price,
            "amount": order.amount,
            "cost": price * order.amount + fee,
            "average": price * order.amount,
            "fee": {"cost": fee, "currency": "EUR"},
        }

        return order_response


if __name__ == "__main__":

    order = Order(
        order_id=uuid4().hex,
        trading_pair="BTC/EUR",
        status="pending",
        side="buy",
        amount=Decimal("0.1"),
        timestamp_tick=timestamp_to_string(pd.Timestamp.now()),
        price_execution=Decimal("32009"),
        cost_execution=Decimal("0.01") * Decimal("32009"),
        timestamp_execution=timestamp_to_string(pd.Timestamp.now()),
    )

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
        backtest=False,
        exchange_name="binance",
        api_key=secrets["api_key"],
        secret_key=secrets["secret_key"],
    )
    settlement = brkr.place_order(order, tick)
    print(settlement)

    # print("\n\n----")
    # orders = brkr.exchange.fetch_orders("BTC/EUR")
    # print(orders)
    # print(len(orders))
