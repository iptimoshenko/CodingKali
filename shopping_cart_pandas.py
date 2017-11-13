import pandas as pd
from pandas import DataFrame as df
import numpy as np
import pandas

prices = df(np.arange(12.).reshape((12,1))+1, index=[['apple', 'apple', 'apple', 'banana', 'banana', 'banana', 'canned_fish',
            'canned_fish', 'canned_fish','dog_food','dog_food','dog_food'],
            [1, 3, 7, 1, 3, 5, 1, 2, 5, 1, 7, 10]], columns=['Price'])

cart = df(np.arange(4.).reshape((4,1))+1, index=['apple',   'banana', 'apple', 'canned_fish'], columns=['quantity'])

def total_bill(cart):
    unique_items = list(cart.index)
    cart['quantity'].groupby(unique_items).sum()
prices[('apple',1)]





