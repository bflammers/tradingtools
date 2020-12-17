
import pandas as pd
import numpy as np

from .data import DataLoader


class Backtest:

    def __init__(self):
        
        self.dl = DataLoader()

    def run(self):
        for row in self.dl.ticker():
            print(row)


if __name__ == "__main__":
    pass
