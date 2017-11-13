import os
os.chdir('C:/Dropbox/coding_docs/Python_Data_Analysis')

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
obj4.isnull()  # as instance methods

# automatic alignment of differently indexed data in arithmetic operations
obj3 + obj4

obj4.name = 'population'
obj4.index.name = 'state'
obj4

# a Series's index can be altered by assignment:
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

# rows retrieval:
frame2.ix['three']
frame2[2:]

frame2['debt']= np.arange(5)
frame2

val = Series([-1.2, -1.5, -1.7], index = ['two', 'four', 'five'])
frame2.debt = val
frame2

frame2['eastern'] = frame2['state']=='Ohio'
frame2

# delete a column
del frame2['eastern']
frame2.columns

# passing nested dictionary to DataFrame:
pop = {'Nevada': {2001: 2.4, 2002: 2.9}, 'Ohio': {2000: 1.5, 2001: 1.7, 2002: 3.6}}
frame3 = DataFrame(pop)
frame3 #  outer keys become columns, inner keys - row indices
frame3.T

# unless an explicit index is specified, the innder ficts are unioned and sorted to form the resulting index.
DataFrame(pop, index = [2003, 2002, 2001])

pdata = {'Ohio': frame3['Ohio'][:-1],
        'Nevada': frame3['Nevada'][:2]}
DataFrame(pdata)

# name attributes can be set to dataframe index and columns:
frame3.index.name = 'year'; frame3.columns.name = 'state'
frame3
frame3.values

## if columns are of different data type, the dtype of the values array will be chosen to accommodate all of the columns:
frame2.values

# Index objects, page 116
# Pandas index objects are responsible for holding the axis labels and other metadata (like the axis name(s))
# index objects are immutable

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

# with dataframe reindex can alter either the (row) index, columns or both.
frame = DataFrame(np.arange(9).reshape((3,3)), index=['a', 'd', 'e'], columns=['Ohio', 'Texas', 'California'])
frame

frame2 = frame.reindex(['a', 'b', 'c', 'd'])
# to reindex columns, use columns keyword
states = ['Texas', 'California', 'Utah']
frame.reindex(columns=states)

# both reindexed in one shot, though interpolation will only apply row-wise: , limit=1 - max of how many emply to fill
frame.reindex(index = ['a', 'b', 'c', 'd', 'e'], method='bfill', columns=states)
frame.ix[['a', 'b', 'c', 'd'], states]

# Dropping entries from an axis
obj = Series(np.arange(5.), index = ['a', 'b', 'c', 'd', 'e'])
obj.drop(['d', 'c'])

data = DataFrame(np.arange(16).reshape(4,4), index =['Texas', 'California', 'Utah', 'New York'], columns = ['one', 'two', 'three', 'four'])
data.drop(['one', 'three'], axis=1)

# Indexing selection and filtering:
data[1:]
data['three']
data['Texas':'California']
data[data['three']>5]
data[['two','one']]
data<5
data[data<5]=0

# to label mix rows and columns:
data.ix[['Texas','California'], ['two', 'three']]
data.ix[3]
data.ix[:'Utah', 'two']
data.icol[3]


# Missing values propagate in arithmetic computations:
s1 = Series([2, 3, 4, 1], index = ['a', 'c', 'd', 'e'])
s2 = Series([2, 3, 4, 5.1, 1], index = ['a', 'c', 'e', 'f', 'g'])
s1+s2

# resulting indices and columns are union of the inputs' ones:
df1 = DataFrame(np.arange(12).reshape((3,4)), columns = list('abcd'))
df2 = DataFrame(np.arange(20).reshape((4,5)), columns = list('abcde'))
df1+df2
# filling non-matching indices:
df2.add(df1, fill_value=0)
df2.mul(df1, fill_value=1)

# adding column from another dataframe and filling the values with 0's:
df1.reindex(columns=df2.columns, fill_value=0)

# operations between dataframe and series are broadcasted:
arr = np.arange(12.).reshape((3,4))
arr - arr[0]

frame = DataFrame(np.arange(12.).reshape((4,3)))

# Function application and mapping
frame = DataFrame(np.random.randn(4,3), columns=list('bde'), index=['Utah', 'Ohio', 'Texas', 'Oregon'])
np.abs(frame)

# function applied to each column or row:
f = lambda x: x.max() - x.min()
frame.apply(f)
frame.apply(f, axis=1)


# function can return a Series with multiple values
def f(x):
        return Series([x.min(), x.max()], index=['min', 'max'])

frame.apply(f)

# Using element-wise functions:
format = lambda x: '%.2f' % x
frame.applymap(format)
frame['e'].map(format)

# p.130 Sorting and ranking
obj = Series(range(4), index=['d', 'a', 'b', 'c'])
obj.sort_index()

frame = DataFrame(np.arange(8).reshape((2,4)), index=['three', 'one'], columns=['d', 'a', 'b', 'c'])
frame.sort_index(axis=1)

# missing values are sorted to the end by default:
obj = Series([4, np.nan, 7, np.nan, -3, 2])
obj.order()

# sort by more than on column/row:
frame=DataFrame({'a':[4,7,-3,2] , 'c':[2,1,0,-1]})
frame.sort_index(by=['a','c'])

# rank breaks ties by assigning each group the mean rank:
obj = Series([7, -5, 7, 4, 2, 0, 4])
obj.rank()
obj.rank(method='first') # according to the order they appear in the data
# descending order:
obj.rank(ascending=False, method='max')


# p.132 Axis indexes with duplicate values:
obj = Series(range(5), index=['a', 'a', 'b', 'b', 'c'])
obj.index.is_unique
# indexing a value with multiple entries returns a series
obj['a']
df = DataFrame(np.random.randn(4,3), index=['a', 'a', 'b', 'b'])
df.ix['b']


# p.133 Summarizing and Computing Descriptive Statistics
df = DataFrame([[1.4, np.nan], [7.1, -4.5], [np.nan, np.nan], [0.75, -1.3]], index=['a', 'b', 'c', 'd'], columns=['one', 'two'])
df.sum(axis=1)
df.mean(axis=1, skipna=False)
# index where the min/max value is located:
df.idxmax(axis=1) # max over columns within a row
df.cumsum(axis=1) # cumcum over columns
# set of descriptives:
df.describe()

# on non-numeric data:
obj=Series(['a', 'a', 'b', 'c']*4)
obj.describe()


# p.136 Correlation and Covariance
import pandas_datareader as web
all_data = {}
for ticker in ['AAPL', 'IBM', 'MSFT', 'GOOG']:
        all_data[ticker]=web.get_data_yahoo(ticker)

price = DataFrame({tic: data['Adj Close'] for tic, data in all_data.iteritems()})
volume = DataFrame({tic: data['Volume'] for tic, data in all_data.iteritems()})

returns = price.pct_change()
returns.tail()
returns.MSFT.corr(returns.IBM)
returns.MSFT.cov(returns.IBM)
returns.corr()
returns.cov()
# correlations between different series:
returns.corrwith(returns.IBM)
# passing a dataframe computes correlations of matching column names:
returns.corrwith(volume, axis=1) # row-wise corrs

# p.137 Unique values, value counts, and membership
obj = Series(['c', 'a', 'b', 'd', 'a', 'b', 'b', 'c', 'c'])
uniques = obj.unique()
uniques.sort()
# frequencies:
obj.value_counts()
pd.value_counts(obj.values, sort=True)
mask=obj.isin(['b', 'c'])
obj[mask]

data = DataFrame({'Qu1': [1,3,4,3,4],
                'Qu2': [2,3,1,2,3],
                'Qu3': [1,5,2,4,4]})
result = data.apply(pd.value_counts).fillna(0)


# p.139 Handling missing data
string_data = Series(['aardvark', 'artichoke', np.nan, 'avocado'])
string_data.isnull()
# None is interpreted as NA
string_data[0]=None
string_data.isnull()
string_data.notnull()
string_data.dropna()

# p.140 Filtering out missing data
from numpy import nan as NA
data = Series([1, NA, 3.5, NA, 7])
data.dropna()
data[data.notnull()]
# only filter out rows that are all NA:
data = DataFrame([[1., 6.5, 3.], [1., NA, NA], [NA, NA, NA], [NA, 6.5, 3.]])
data.dropna(how='all')
data[4]=NA
data.dropna(axis=1, how='all')
# min threshold for present data to be kept:
df = DataFrame(np.random.randn(7, 3))
df.ix[:4, 1] = NA; df.ix[:2, 2] = NA
df.dropna(thresh=2)

# p.142 Filling in Missing Data
# fill with different values in each column
df.fillna({1: 0.5, 2: -1})
# to modify the existing object instead of returning a new one:
_ = df.fillna(0, inplace=True)
df
#
df = DataFrame(np.random.randn(6, 3))
df.ix[2:, 1] = NA; df.ix[4:, 2] = NA
df.fillna(method='ffill', limit=2, axis=1)
df.fillna(df.mean())
df.fillna(value=154)

# p.143 Hierarchical Indexing
data = Series(np.random.randn(10), index=[['a', 'a', 'a', 'b', 'b', 'b', 'c', 'c', 'd', 'd'], [1, 2, 3, 1, 2, 3, 1, 2, 2, 3]])
# partial indexing
data['b':'c']
data.ix[['b', 'd']]
# indexing an inner level:
data[:, 2]
data.unstack()
data.unstack().stack()
# with a dataframe, either axis can gave a hierarchical index:
frame = DataFrame(np.arange(12).reshape((4,3)), index=[['a', 'a', 'b', 'b'], [1, 2, 1, 2]], columns=[['Ohio', 'Ohio', 'Colorado'], ['Green', 'Red', 'Green']])
frame
frame.index.names = ['key1', 'key2']
frame.columns.names=['state', 'colour']
frame

# p.146 Reordering and sorting levels
frame.swaplevel('key1', 'key2')
frame.swaplevel(0,1).sortlevel(0)

# p.147 Summary statistics by level:
frame.sum(level='key2')
frame.sum(level='colour', axis=1)

# Using a dataframe's columns:
frame = DataFrame({'a': range(7), 'b': range(7, 0, -1), 'c': ['one', 'one', 'one', 'two', 'two', 'two', 'two'],
                   'd': [0, 1, 2, 0, 1, 2, 3]})
frame2 = frame.set_index(['c', 'd'])
# to keep columns one is indexing by:
frame.set_index(['c', 'd'], drop=False)
# to revert setting indices:
frame2.reset_index()

# p.148 Integer Indexing
ser = Series(np.arange(3.))
ser.ix[:1]
ser3 = Series(range(3), index = [-5, 1, 3])
ser3.iget_value(2)
frame = DataFrame(np.arange(6).reshape(3,2), index=[2, 0, 1])
frame.irow(0)

# p.149 Panel data
import pandas_datareader as web
pdata = pd.Panel(dict((stk, web.get_data_yahoo(stk)) for stk in ['AAPL', 'IBM', 'MSFT', 'GOOG']))
pdata = pdata.swapaxes('items', 'minor')
pdata['Adj Close']
# ix-based label indexing generalizes to 3 dimensions
pdata.ix['Adj Close', '6/1/2017':, :]
# stacked dataframe form:
stacked = pdata.ix[:, '5/30/2012':, :].to_frame()
# the inverse of to_frame:
stacked.to_panel()

# test lines