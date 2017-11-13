import os
os.chdir('C:/Dropbox/coding_docs/Python_Data_Analysis')
#########################################################
import json
import numpy as np
import matplotlib
from pandas import DataFrame, Series
import pandas as pd

### every array has a shape and a datatype:
data1 = [6, 7.5, 8, 0, 1]
arr1 = np.array(data1)
arr1.shape
arr1.dtype


# one of the ways to create a matrix
data2 = [[1,2,3,4], [5,4,6,7]]
arr2 = np.array(data2)
arr2.shape
arr2.ndim

### create array of 0
np.zeros((3,6))
np.zeros(10)

# empty array:
np.empty((2,3,2))

# range in array:
np.arange(15)

## to produce array of 1's with same shape and dtype:
np.ones_like(arr2)
np.eye(5)
np.identity(5)

arr1 = np.array([1,2,3], dtype = np.float64)
arr2 = np.array([1,2,3], dtype = np.int32)

# convert array to different data type:
float_arr2 = arr2.astype(np.float64)

# use another array dtype to convert:
int_array = np.arange(10)
calibers = np.array([.7, .4,.065, .4], dtype = np.float64)
int_array.astype(calibers.dtype)

### array multiplication happens elementwise:
arr = np.array([[1.,2., 3.], [4.,5.,6.]])
arr*arr
arr*0.5

# Slicing arrays
arr = np.arange(10)
arr[5:8]
# to broadcast value to view points in array. Array slices are views on the original array:
arr[5:8] = 12
arr_slice = arr[5:8]
arr_slice[1] = 12345
arr

arr_slice[:] = 64
arr

## to copy the slice without changing the original
arr = np.arange(10)
arr_sl = arr[5:8].copy()
arr_sl[:] = 64
arr

### individual elements can be accessed recurcively:
arr = np.array([[1.,2., 3.], [4.,5.,6.]])
arr[0][2]
arr[0, 2]

arr3d = np.array([[[1,2, 3], [4,5,6]], [[7,8,9],[10,11,12]]])
arr3d[1]

# both scalar values and arrays can ve assigned to a 2-dimensional part of an array:
old_values = arr3d[0].copy()
arr3d[0]=42
arr3d[0]

arr3d[0] = old_values

arr2 = np.array([[1.,2., 3.], [4.,5.,6.], [1.,7, 3.]])
arr2[:2]
arr2[:2, 1:]
arr2[:, :1]

## assing to a slice:
arr2[:2, 1:] = 1
arr2

names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will','Joe',  'Joe'])
data = np.random.randn(7,4)
names == 'Bob' # compare all entries in array 

data[names == 'Bob'] # boolean array must be of the same length as the axis it's indexing
# mix and match Boolean arrays with slicing:
data[names == 'Bob', 2:]

## Boolean indexing
# to use opposite of Boolean array, exclude certain values:
data[-(names=='Bob')]

### combining Boolean:
mask = (names == 'Bob') | (names == 'Will') 
### Selection by Boolean indexing always creates a copy of an array
data[mask]

# set all negative values to 0
data[data<0] = 0

## fancy indexing: indexing using integer arrays
## NNNNNBBBB always copies a data into a new array
arr = np.empty((8,4))
for i in range(8):
    arr[i] = i
arr

## to select a subset of rows in particular order, pass an ndarray or a list:
arr[[4,3,0,6]]

## select rows from the bottom by using negative indices:
arr[[-3, -5, -7]]

arr = np.arange(32).reshape((8,4))
arr[[1,5,7,2],[0,3,1,2]]

# to get a matrix:
arr[[1,5,7,2]][:,[0,3,1,2]]

## otherwise np.ix function converts two 1D integer arrays into a indexer that selects the square region:
arr[np.ix_([1,5,7,2],[0,3,1,2])]

## p 89 Transposing returns the view of underlying data
arr.T

## inner matrix product:
np.dot(arr.T, arr)

## for higher dimensional arrays:
arr = np.arange(16).reshape((2,2,4))
arr
arr.transpose((1,0,2))

## swap axis:
arr.swapaxes(1,2)


#### page 91
## Universal Functions: Element-wise Array functions
## ufunc:fast vectorized wrappers for simple functions
arr = np.arange(10)
np.sqrt(arr)

x = np.random.randn(8)
y = np.random.randn(8)
np.maximum(x,y) # element-wise maximum

arr = randn(7)*5
arr
np.modf(arr) # fractio and integer part of a floating point array

isnan(arr)

log1p(arr) ## log(1+x)
floor(arr)
ceil(arr) # computes the ceiling

isfinite(arr)
isinf(arr)

logical_not(arr) # the truth value of not x. Equivalent to -arr.

floor_divide(arr, arr+1)
power(arr, arr+1)
fmax(arr+2, arr) # ifnores nan, as well as fmin

copysign(arr+3, arr) # copies sign of values in second argument to values in first argument

logical_xor(arr+2, arr)

points = np.arange(-5, 5, 0.01) # 1000 equally spaced points
xs, ys = np.meshgrid(points, points)
ys
ys.shape

import matplotlib.pyplot as plt
z = np.sqrt(xs**2 + ys**2)
plt.imshow(z, cmap=plt.cm.gray); plt.colorbar()
plt.title("Image plot of $\sqrt{x^2 + y^2}$ for a grid of values")

###############
##### Conditional logic as array operations, page 94

result = []
for i in range(n):
    if cond1[i] and cond2[i]:
        result.append(0)
    elif cond1[i]:
        result.append[1]
    elif cond2[i]:
        result.append(2)
    else:
        result.append(3)
        
#using the nested where expression:
np.where(cond1 & cond2, 0, np.where(cond1, 1, np.where(cond2, 2, 3)))

cond1 = False
cond2 = True
1*(cond1 & -cond2)+2*(cond2 & -cond1) + 3* -(cond1| cond2)

## Mathematical and statistical methods, page 96
arr = np.random.randn(5,4)
arr
arr.mean(axis=0) ## axis argument is optional
arr.sum(axis=1)
arr.sum(0)
arr.std()

arr = np.array([[0,1,2], [3,4,5], [6,7,8]])
arr.cumsum(0)
arr.cumprod(1)
arr.argmin(0) # indices og minimum elements

# Methods for Boolean Arrays:
arr = randn(100)
(arr>0).sum() # counting number of positives
# is any in the array true:
bools = np.array([False, False, True, False])
bools.any()
bools.all() # are all true?

real = np.array([3.4, 55, 1.4, 6, 0 , -2.2])
real.any() ## non-zero elements evaluate to true
real.all()

arr = randn(9)
arr.sort()
arr

arr = randn(5,3)  ## returns a sorted copy of an array instead of modifying the array in place
arr.sort(0)
arr

large_arr = randn(1000)
large_arr.sort()
large_arr[int(0.05*len(large_arr))] # 5% quantile

## Unique and other set logic:
names = np.array(['Bob', 'Joe', 'Joe', 'Will', 'Bob', 'Will'])
## sorted unique values in array:
np.unique(names)

values = np.array([6,5,8,4,98,2,97,7])
np.in1d(values, [2,5,7, 4])
np.intersect1d(values, [2,5,7, 4]) # sprted common elements
np.setdiff1d(values, [2,5,7, 4]) # set difference
np.setxor1d(values, [2,5,7, 4]) # set symmetric differences: elements that are in either of the arrays, but not in both

## Storing arrays on disk in binary format:
## np.save and np.load
arr = np.arange(10)
np.save('some_array', arr)
np.load('some_array.npy')

# saving multiple arrays in zip archive:
np.savez('array_achive.npz', a=arr, b=arr)
arch = np.load('array_achive.npz')
arch['a']

##  saving and loading text files
arr = np.loadtxt('data/ch04/array_ex.txt', delimiter = ',')
np.savetxt('data/ch04/saved_array_ex.txt', arr, delimiter = ',')

### Linear algebra, page 101
x = np.array([[1,2,3],[4,5,6]])
y = np.array([[1,6], [-4, 7] , [7,1]])
x.dot(y) 
# equivalent:
np.dot(x,y)

np.dot(x, np.ones(3))
np.dot(np.ones(3),y)

from numpy.linalg import inv, qr
X = randn(5,5)
mat = X.T.dot(X)
inv(mat)
mat.dot(inv(mat))
q, r = qr(mat)
r
q
eig(mat)
svd(mat)


### Random Number Generation:
samples =np.random.normal(size=(4,4))
from random import normalvariate
N = 1000000
%timeit samples = [normalvariate(0,1) for _ in xrange(N)]
%timeit np.random.normal(size=N)

## RAndom walks
import random
position = 0
walk = [position]
steps = 1000
for i in xrange(steps):
    step = 1 if random.randint(0,1) else -1
    position += step
    walk.append(position)
plot(walk)

nsteps = 1000
draws = np.random.randint(0,2,size = nsteps)
steps = np.where(draws>0, 1, -1)
walk = steps.cumsum()
walk.max()
plot(walk)

(np.abs(walk)>=10).argmax() ### returns the first index of the max value in the boolean array, which is True


## Multiple random walks:
nwalks = 20
nsteps = 1000
draws = np.random.randint(0,2,size = (nwalks, nsteps))
steps = np.where(draws>0, 1, -1)
walks = steps.cumsum(1)
walks.max()
plot(walks) 

hits30 = (np.abs(walks)>=30).any(1)
hits30
crossing_times = (np.abs(walks[hits30])>=30).argmax(1)
first_crossing_time = (np.abs(walks[hits30])>=30).argmax()