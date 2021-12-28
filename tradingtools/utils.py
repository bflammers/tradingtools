import sys
import signal
import logging

from enum import Enum
from typing import Tuple
from dataclasses import dataclass
from decimal import Decimal
from time import time
from uuid import uuid4


try:
    # unix / macos only
    from signal import SIGHUP

    SIGNALS = (signal.SIGABRT, signal.SIGINT, signal.SIGTERM, SIGHUP)
except ImportError:
    SIGNALS = (signal.SIGABRT, signal.SIGINT, signal.SIGTERM)

logger = logging.getLogger(__name__)


class Colors(Enum):
    blue = "\033[94m"
    cyan = "\033[96m"
    green = "\033[92m"
    grey = "\x1b[38;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"


class ColoredLogFormatter(logging.Formatter):

    format = (
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)"
    )

    FORMATS = {
        logging.DEBUG: Colors.cyan.value + format + Colors.reset.value,
        logging.INFO: Colors.grey.value + format + Colors.reset.value,
        logging.WARNING: Colors.yellow.value + format + Colors.reset.value,
        logging.ERROR: Colors.red.value + format + Colors.reset.value,
        logging.CRITICAL: Colors.bold_red.value + format + Colors.reset.value,
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


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
            return base, quote
        except IndexError:
            pass

    message = f"[split_pair] Not able to split {pair} on {seperators}"
    logger.error(message)
    raise Exception(message)


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
    quantity: Decimal
    type: str
    status: str = "open"
    price: Decimal = None
    timestamp_created: float = None
    order_id: str = None
    # Settlement
    price_settlement: Decimal = None
    cost_settlement: Decimal = None
    timestamp_settlement: float = None
    filled_quantity: Decimal = None
    exchange_order_id: str = None
    fee: Decimal = None
    fee_currency: str = None
    trades: list = None

    def __post_init__(self):

        if self.timestamp_created is None:
            self.timestamp_created = time()

        if self.order_id is None:
            self.order_id = uuid4().hex

        if self.type == "market" and self.price is not None:
            logger.debug("[Order] order type is market and price is not None")

        if self.type == "limit" and self.price is None:
            logger.error("[Order] order type is limit and price is None")

    def update(
        self,
        price: Decimal = None,
        cost: Decimal = None,
        timestamp: float = None,
        filled_quantity: Decimal = None,
        exchange_order_id: str = None,
        fee: Decimal = None,
        fee_currency: str = None,
        trades: list = None,
        status: str = "settled",
    ) -> None:

        self.price_settlement = price
        self.cost_settlement = cost
        self.timestamp_settlement = timestamp
        self.filled_quantity = filled_quantity
        self.exchange_order_id = exchange_order_id
        self.fee = fee
        self.fee_currency = fee_currency
        self.trades = trades
        self.status = status
