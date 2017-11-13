import pandas as pd
from pandas import DataFrame as df
import numpy as np
import pandas

prices = df({'price': np.arange(12.)+1, 'product': ['apple', 'apple', 'apple', 'banana', 'banana', 'banana', 'canned_fish',
            'canned_fish', 'canned_fish','dog_food','dog_food','dog_food'],
             'quantity_tresh': [1, 3, 7, 1, 3, 5, 1, 2, 5, 1, 7, 10]})
prices = prices.set_index(['product', 'quantity_tresh'])

cart = df(np.arange(4.).reshape((4,1))+1, index=['apple',   'banana', 'apple', 'canned_fish'], columns=['quantity'])

def get_cart_items_and_quantities(cart):
    unique_items = list(cart.index.unique())
    cart2 = df(index=unique_items, columns=['quantity'])
    for item in unique_items:
        if sum(item == cart.index) > 1:
            cart2['quantity'].loc[item] = sum(cart['quantity'].loc[item])
        else:
            cart2['quantity'].loc[item] = cart['quantity'].loc[item]
    return cart2


def total_cost(cart, prices):
    cart = get_cart_items_and_quantities(cart)
    print(cart)
    for item in cart.index:
        thresholds = list(prices.loc[item].index)
        thresh_1 = max(i for i in thresholds if i <= cart['quantity'].loc[item])
        cart['quantity'].loc[item] % thresholds
        # check if quantitiy in cart is greater than all offer values. If so, total cost = higher quant*multiple +
        # mod(cart quantity/higher offer)
        print(item, thresh_1)

    return (item, thresh_1)






