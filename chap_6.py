import pandas_datareader as web
import json
import numpy as np
import matplotlib
from pandas import DataFrame, Series
import pandas as pd

!cat data/ch06/ex1.csv # prints out raw content of a file
df = pd.read_csv('data/ch06/ex1.csv')
pd.read_table('data/ch06/ex1.csv', sep=',')

pd.read_csv('data/ch06/ex2.csv', header=None)
pd.read_csv('data/ch06/ex2.csv', names=['a', 'b', 'c', 'd', 'message'])

names=['a', 'b', 'c', 'd', 'message']
pd.read_csv('data/ch06/ex2.csv', names=names, index_col='message')

parsed = pd.read_csv('data/ch06/csv_mindex.csv', index_col=['key1', 'key2'])
parsed

# passing a regular expression as a delimiter:
list(open('data/ch06/ex3.txt'))
result = pd.read_table('data/ch06/ex3.txt', sep='\s+')
result
# skip rows
pd.read_csv('data/ch06/ex4.csv', skiprows=[0, 2, 3])

result = pd.read_csv('data/ch06/ex5.csv')
pd.isnull(result)

result = pd.read_csv('data/ch06/ex5.csv', na_values=['NULL'])
sentinels = {'message':['foo', 'NA'], 'something':['two']}
result = pd.read_csv('data/ch06/ex5.csv', na_values=sentinels)
result




# p.169 Using HDF5 Format
# Hierarchical data format
store = pd.HDFStore('mydata.h5')
store['obj1'] = frame
# store['obj1_col'] = frame['a']
store