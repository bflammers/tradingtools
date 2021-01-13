import pandas as pd
import numpy as np

from pathlib import Path
from decimal import Decimal

from tradingtools.portfolio import Portfolio


def test_init():

    starting_capital = Decimal(100)

    pf = Portfolio(starting_capital=starting_capital, 
                   results_parent_dir="./runs/unit_tests", 
                   verbose=False)

    assert pf._tick_ohlcv_path.is_file()
    assert pf._opt_positions_path.is_file()
    assert pf._orders_path.is_file()
    assert pf._settlements_path.is_file()

    assert pf._starting_capital == starting_capital
    assert pf._unallocated_capital == starting_capital
    assert pf._reserved_capital == Decimal(0)


def test_update():

    starting_capital = Decimal(100)

    pf = Portfolio(starting_capital=starting_capital, 
                   results_parent_dir="./runs/unit_tests", 
                   verbose=True)

    


    


if __name__ == "__main__":

    test_init()
    test_update()
