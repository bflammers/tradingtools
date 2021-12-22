import sys
import signal

from logging import getLogger
from typing import Tuple
from dataclasses import dataclass
from decimal import Decimal
from datetime import datetime
from uuid import uuid4


try:
    # unix / macos only
    from signal import SIGHUP

    SIGNALS = (signal.SIGABRT, signal.SIGINT, signal.SIGTERM, SIGHUP)
except ImportError:
    SIGNALS = (signal.SIGABRT, signal.SIGINT, signal.SIGTERM)

logger = getLogger(__name__)


def setup_signal_handlers(loop):
    """
    This must be run from the loop in the main thread
    """

    def handle_stop_signals(*args):
        raise SystemExit

    if sys.platform.startswith("win"):
        # NOTE: asyncio loop.add_signal_handler() not supported on windows
        for sig in SIGNALS:
            signal.signal(sig, handle_stop_signals)
    else:
        for sig in SIGNALS:
            loop.add_signal_handler(sig, handle_stop_signals)


def split_pair(pair: str) -> Tuple:

    seperators = ["-", "/"]
    for sep in seperators:

        try:
            splitted = pair.split(sep)
            base, quote = splitted[0], splitted[1]
        except IndexError:
            pass

        if len(splitted) == 2:
            return base, quote

    logger.error(f"[split_pair] Not able to split {pair} on {seperators}")


def length_string_to_seconds(length: str) -> int:

    # Extract quantity and unit
    quantity = int(length[:-1])
    unit = length[-1]

    # Determine multiplier based on unit
    try:
        multiplier = {"S": 1, "M": 60, "H": 3600, "D": 86400}[unit]
    except KeyError:
        raise ValueError(
            f"[length_string_to_seconds] length argument {length} unit {unit} not supported"
        )

    return quantity * multiplier


@dataclass
class Order:

    # Execution
    symbol: str
    side: str
    amount: Decimal
    type: str
    status: str = "open"
    price: Decimal = None
    timestamp_created: datetime = datetime.now()
    order_id: str = uuid4().hex
    # Settlement
    price_settlement: Decimal = None
    amount_settlement: Decimal = None
    timestamp_settlement: datetime = None
    exchange_order_id: str = None
    fee: Decimal = None
    fee_currency: str = None

    def __post_init__(self):

        if type == "market" and self.price is not None:
            logger.warning("[Broker] order type is market and price is not None")

    def settle(
        self,
        price: Decimal = None,
        amount: Decimal = None,
        timestamp: datetime = None,
        exchange_order_id: str = None,
        fee: Decimal = None,
        fee_currency: str = None,
    ) -> None:

        self.price_settlement = price
        self.amount_settlement = amount
        self.timestamp_settlement = timestamp
        self.exchange_order_id = exchange_order_id
        self.fee = fee
        self.fee_currency = fee_currency
        self.status = "settled"


# if __name__ == "__main__":

#     print(split_pair("BTC-EUR"))
#     print(split_pair("BTC/EUR"))

#     p = Prices(time_tol_sec=0)
#     p.update("BTC/EUR", 10)
#     print(p.data)
#     print(p.get("BTC-EUR"))
#     p.update("BTC-EUR", 12)
#     print(p.get("BTC/EUR"))

#     p.update("BTC/USD", 5)
#     p.update("ETH/EUR", 6)

#     data = p.data
#     print(data)

#     p.get("BTC-AAA")
