
import pandas as pd
import numpy as np


class DataLoader:

    def __init__(self):
        self.df = pd.DataFrame({'a': [1,2,3], 'b': [4,5,6]})
        self.n = self.df.shape[0]

    def ticker(self):
        for index, row in self.df.iterrows():
            yield row

    def get_df(self):
        return self.df

if __name__ == "__main__":
    pass