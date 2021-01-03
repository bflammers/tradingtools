from decimal import Decimal
import pandas as pd
import warnings


def warning_on_one_line(message, category, filename, lineno, file=None, line=None):
    return "%s:%s: %s: %s\n" % (filename, lineno, category.__name__, message)


warnings.formatwarning = warning_on_one_line


class colors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


def color_number_sign(x: Decimal, decimals: int = 3, offset: float = 0) -> str:
    if (x - offset) > 0:
        return f"{colors.OKGREEN}+{x:.{decimals}f}{colors.ENDC}"
    else:
        return f"{colors.FAIL}{x:.{decimals}f}{colors.ENDC}"


def print_item(
    currency: str,
    value: Decimal,
    profit: Decimal = None,
    profit_percentage: Decimal = None,
    n_orders: int = None,
    n_open_orders: int = None,
) -> str:

    value_colored = color_number_sign(value, decimals=2)
    out = f"{currency} {value_colored}"

    if profit is not None:
        profit_colored = color_number_sign(profit)
        out += f" / {profit_colored} profit"

    if profit_percentage is not None:
        profit_percentage_colored = color_number_sign(profit_percentage)
        out += f" / {profit_percentage_colored} % profit"

    if n_orders is not None:
        out += f" / {n_orders} orders"

    if n_open_orders is not None:
        out += f" / {n_open_orders} open orders"

    return out


_strftime_format = "%Y-%m-%d %H:%M:%S.%f"


def timestamp_to_string(ts: pd.Timestamp) -> str:
    return ts.strftime(_strftime_format)


def string_to_timestamp(ts_string: str) -> pd.Timestamp:
    return pd.Timestamp(ts_string)


def extract_prices(tick: list, price_type: str = "close") -> dict:
    return {t["symbol"]: Decimal(t[price_type]) for t in tick if price_type in t}
