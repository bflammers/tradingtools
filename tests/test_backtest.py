
import pandas as pd

from tradingtools import backtesting

def test_backtest():

    bt = backtesting.Backtest()
    bt.run()


if __name__ == "__main__":

    bt = backtesting.Backtest()
    bt.run()
