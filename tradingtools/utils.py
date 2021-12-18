
from logging import getLogger
from typing import Tuple

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
