from decimal import Decimal
from time import time
from logging import getLogger
from typing import Dict, Tuple

logger = getLogger(__name__)


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


class Prices:

    time_tol_sec: float
    data: Dict[str, Dict[str, dict]] = {}

    def __init__(self, time_tol_sec: float = 120.0) -> None:
        self.time_tol_sec = time_tol_sec

    def clear(self) -> None:
        self.data = {}

    def _safe_get(self, base: str, quote: str) -> dict:

        try:
            base_dict = self.data[base]
        except KeyError:
            self.data[base] = {}
            base_dict = self.data[base]

        try:
            quote_dict = base_dict[quote]
        except KeyError:
            self.data[base][quote] = {}
            quote_dict = self.data[base][quote]

        return quote_dict

    def _safe_set(self, base: str, quote: str, price: Decimal) -> dict:

        _ = self._safe_get(base, quote)
        self.data[base][quote] = {
            "price": Decimal(price),
            "update_time": time(),
            "pair": f"{base}/{quote}",
        }

    def update(self, pair, price):
        base, quote = split_pair(pair)
        self._safe_set(base, quote, price)

    def get(self, pair):
        base, quote = split_pair(pair)
        price = self._safe_get(base, quote)

        if price:
            time_diff = time() - price["update_time"]
            if time_diff > self.time_tol_sec:
                logger.warning(
                    f"[Prices] asset_name {pair} not updated for {time_diff} seconds"
                )
        else:
            logger.warning(f"[Prices] asset_name {pair} not yet updated")
            return None

        return price["price"]


if __name__ == "__main__":

    print(split_pair("BTC-EUR"))
    print(split_pair("BTC/EUR"))

    p = Prices(time_tol_sec=0)
    p.update("BTC/EUR", 10)
    print(p.data)
    print(p.get("BTC-EUR"))
    p.update("BTC-EUR", 12)
    print(p.get("BTC/EUR"))

    p.update("BTC/USD", 5)
    p.update("ETH/EUR", 6)

    data = p.data
    print(data)

    p.get("BTC-AAA")
