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


_strftime_format = "%Y-%m-%d %H:%M:%S.%f"


def timestamp_to_string(ts: pd.Timestamp) -> str:
    return ts.strftime(_strftime_format)


def string_to_timestamp(ts_string: str) -> pd.Timestamp:
    return pd.Timestamp(ts_string)


def extract_prices(tick: list, price_type: str = "close") -> dict:
    return {t["symbol"]: t[price_type] for t in tick if price_type in t}
