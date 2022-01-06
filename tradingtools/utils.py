import sys
import signal
import logging


from enum import Enum
from typing import Tuple
from dataclasses import dataclass
from decimal import Decimal
from time import time
from uuid import uuid4
from math import log10


try:
    # unix / macos only
    from signal import SIGHUP

    SIGNALS = (signal.SIGABRT, signal.SIGINT, signal.SIGTERM, SIGHUP)
except ImportError:
    SIGNALS = (signal.SIGABRT, signal.SIGINT, signal.SIGTERM)

logger = logging.getLogger(__name__)


class RunType(Enum):
    backtest = "backtest"
    dry_run = "dry_run"
    live = "live"


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

    format = "%(asctime)s - %(levelname)s - %(message)s (%(name)s:%(lineno)d)"

    DEFAULT = logging.Formatter(format)

    FORMATTERS = {
        logging.DEBUG: logging.Formatter(
            Colors.cyan.value + format + Colors.reset.value
        ),
        logging.INFO: logging.Formatter(
            Colors.grey.value + format + Colors.reset.value
        ),
        logging.WARNING: logging.Formatter(
            Colors.yellow.value + format + Colors.reset.value
        ),
        logging.ERROR: logging.Formatter(
            Colors.red.value + format + Colors.reset.value
        ),
        logging.CRITICAL: logging.Formatter(
            Colors.bold_red.value + format + Colors.reset.value
        ),
    }

    def format(self, record):
        formatter = self.FORMATTERS.get(record.levelno, self.DEFAULT)
        return formatter.format(record)


def round_decimal(value: Decimal, precision: int = 6) -> Decimal:

    if precision < 0:
        message = "[round_decimal] precision cannot be negative"
        logger.error(message)
        raise ValueError(message)

    q = Decimal(f"{'0.' if precision > 0 else ''}{'0'*(precision-1)}1")
    return value.quantize(q)


def float_to_decimal(value: float, precision: int = 6) -> Decimal:

    if value is None:
        return value

    return round_decimal(Decimal(value), precision)


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


def split_interval(interval: str) -> Tuple[int, str]:
    # Extract quantity and unit
    quantity = int(interval[:-1])
    unit = interval[-1]
    return quantity, unit


def interval_to_seconds(interval: str) -> int:

    quantity, unit = split_interval(interval)
    # Determine multiplier based on unit
    try:
        multiplier = {"S": 1, "M": 60, "H": 3600, "D": 86400}[unit]
    except KeyError:
        raise ValueError(
            f"[length_string_to_seconds] length argument {interval} unit {unit} not supported"
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
    filled_quantity: Decimal = Decimal("0")
    exchange_order_id: str = None
    fee: Decimal = None
    fee_currency: str = None
    trades: list = None

    def __post_init__(self):

        if self.price is not None and self.price < Decimal("0"):
            message = f"[Order] order cannot have a negative price"
            logger.error(message)
            raise ValueError(message)

        if self.quantity < Decimal("0"):
            message = f"[Order] order cannot have a negative quantity"
            logger.error(message)
            raise ValueError(message)

        if self.timestamp_created is None:
            self.timestamp_created = time()

        if self.order_id is None:
            self.order_id = uuid4().hex

        if self.type == "market" and self.price is not None:
            logger.debug("[Order] order type is market and price is not None")

        if self.type == "limit" and self.price is None:
            message = "[Order] order type is limit and price is None"
            logger.error(message)
            raise Exception(message)

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
