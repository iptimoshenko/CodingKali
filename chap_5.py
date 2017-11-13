import os
os.chdir('C:/Dropbox/coding_docs/Python_Data_Analysis')
#########################################################
import json
import numpy as np
import matplotlib
from pandas import DataFrame, Series
import pandas as pd

obj = Series([4,5, -7, 3])
obj.values
obj.index

obj2 = Series([4,5, -7, 3], index = ['d', 'b', 'a', 'c'])
obj2
obj2.index
obj2[['a', 'b', 'd']]

obj2[obj2 > 0]
obj2*2
np.exp(obj2)
# series can be thought of as a fixed-length ordered dictionary,
# hence many functions that take dictionaries will be able to work with Series 
'b' in obj2
sdata = {'Ohio' : '35000', 'Texas' : '71000', 'Oregon' : '16000', 'Utah' : '5000'}
obj3 = Series(sdata)
obj3
obj3.index

states = ['California', 'Ohio', 'Oregon', 'Texas']
obj4 = Series(sdata, index = states)
obj4
pd.isnull(obj4)
pd.notnull(obj4)
obj4.isnull()  ## as instance methods

# automatic alignment of differently indexed data in arithmetic operations
obj3 + obj4

obj4.name = 'population'
obj4.index.name = 'state'
obj4

## a Series's index can be altered by assignment:
obj.index = ['Bob', 'Irina', 'Mateo', 'Radu']
obj

data = {'state': ['Ohio', 'Ohio','Ohio','Nevada', 'Nevada'],
        'year': [2000, 2001, 2002, 2001, 2002],
        'pop': [1.5,1.7, 3.6, 2.4, 2.9]}
frame  = DataFrame(data) # gives columns in sorted order unless a sequence of columns is specified.
DataFrame(data, columns=['year', 'state', 'pop'])
frame2 = DataFrame(data, columns=['year', 'state', 'pop', 'debt'], index = ['one', 'two', 'three', 'four', 'five'])
frame2
# 2 ways to retrieve a column:
frame2['state']
frame2.year

## rows retrieval:
frame2.ix['three']
frame2[2:]

frame2['debt']= np.arange(5)
frame2

val = Series([-1.2, -1.5, -1.7], index = ['two', 'four', 'five'])
frame2.debt = val
frame2

frame2['eastern'] = frame2['state']=='Ohio'
frame2

## delete a column
del frame2['eastern']
frame2.columns

### passing nested dictionary to DataFrame:
pop = {'Nevada': {2001: 2.4, 2002: 2.9}, 'Ohio': {2000: 1.5, 2001: 1.7, 2002: 3.6}}
frame3 = DataFrame(pop)
frame3 #  outer keys become columns, inner keys - row indices
frame3.T

# unless an explicit index is specified, the innder ficts are unioned and sorted to form the resulting index.
DataFrame(pop, index = [2003, 2002, 2001])

pdata = {'Ohio': frame3['Ohio'][:-1],
        'Nevada': frame3['Nevada'][:2]}
DataFrame(pdata)

## name attributes can be set to dataframe index and columns:
frame3.index.name = 'year'; frame3.columns.name = 'state'
frame3
frame3.values

## if columns are of different data type, the dtype of the values array will be chosen to accommodate all of the columns:
frame2.values

### Index objects, page 116
## Pandas index objects are responsible for holding the axis labels and other metadata (like the axis name(s))
## index objects are immutable

obj = Series(range(3), index = ['a', 'b', 'c'])
index = obj.index
index[1:]

index = pd.DatetimeIndex(np.arange(3))
obj2 = Series([1.2, 20, 3.6], index = index)
obj2.index is index

'Ohio' in frame3.columns                                                                                                                                           
2003 in frame3.index

obj = Series([1.2, 20, 3.6, -5.6], index = ['d', 'b', 'a', 'c'])
obj2 = obj.reindex( ['a', 'b', 'c', 'd', 'e'], fill_value=1)
obj2

obj3 = Series(['blue', 'purple', 'yellow'], index=[0, 2, 4])
obj3.reindex(range(6), method='bfill') # or ffill

## with dataframe reindex can alter either the (row) index, columns or both.
frame = DataFrame(np.arange(9).reshape((3,3)), index=['a', 'd', 'e'], columns=['Ohio', 'Texas', 'California'])
frame

frame2 = frame.reindex(['a', 'b', 'c', 'd'])
# to reindex columns, use columns keyword
states = ['Texas', 'California', 'Utah']
frame.reindex(columns=states)

### both reindexed in one shot, though interpolation will only apply row-wise: , limit=1 - max of how many emply to fill
frame.reindex(index = ['a', 'b', 'c', 'd', 'e'], method='bfill', columns=states)
frame.ix[['a', 'b', 'c', 'd'], states]

## Dropping entries from an axis
obj = Series(np.arange(5.), index = ['a', 'b', 'c', 'd', 'e'])
obj.drop(['d', 'c'])

data = DataFrame(np.arange(16).reshape(4,4), index =['Texas', 'California', 'Utah', 'New York'], columns = ['one', 'two', 'three', 'four'])
data.drop(['one', 'three'], axis=1)

## Indexing selection and filtering:
data[1:]
data['three']
data['Texas':'California']
data[data['three']>5]
data[['two','one']]
data<5
data[data<5]=0

## to label mix rows and columns:
data.ix[['Texas','California'], ['two', 'three']]
data.ix[3]
data.ix[:'Utah', 'two']
data.icol[3]
