import os
os.chdir('C:/Dropbox/coding_docs/Python_Data_Analysis')
#########################################################
import json
import numpy as np
import matplotlib
from pandas import DataFrame, Series
import pandas as pd
import datetime

source_file = open('usagov.txt')
records = [json.loads(line) for line in source_file] #json.loads(line)


def get_counts(sequence):
    counts = dict()
    for x in sequence:
        if x in counts.keys():
            counts[x] = counts[x] + 1
        else:
            counts[x] = 1
    return counts
get_counts(records)

def top_ten_counts(count_dict, n=10):
    value_key_pairs = [(count, tz) for tz, count in count_dict.items()]
    value_key_pairs.sort()
    return value_key_pairs[-n:]

time_zones = [rec['tz'] for rec in records ]
from collections import Counter
counts = Counter(time_zones)
counts.most_common(10)

from pandas import DataFrame, Series
import pandas as pd
frame = DataFrame(records)
frame.info()

frame['tz']
# frequency table
tz_counts = frame['tz'].value_counts()
tz_counts[:10]

clean_tz = frame['tz'].fillna('Missing')
clean_tz[clean_tz == ''] = 'Unknown'
tz_counts = clean_tz.value_counts()
tz_counts[:10].plot(kind = 'barh', rot = 0)

# splits line and takes the first value in list
results = Series([x.split()[0] for x in frame.a.dropna()])
results.value_counts()[:10]

cframe = frame[frame.a.notnull()]
operating_system = np.where(cframe['a'].str.contains('Windows'), 'Windows', 'Not Windows')
operating_system[:5]
## grouping data by 2 columns
by_tz_os = cframe.groupby(['tz', operating_system])

## crosstab, shape into table using unstack:
agg_counts = by_tz_os.size().unstack().fillna(0)
agg_counts[:10]

# use to sort in ascending order
indexer = agg_counts.sum(1).argsort()
indexer[:10]

# last 10 rows:
count_subset = agg_counts.take(indexer)[-10:]
count_subset

# stacked bar chart:
count_subset.plot(kind='barh', stacked = True)

# normalised bars:
normed_subset = count_subset.div(count_subset.sum(1), axis = 0)
normed_subset.plot(kind = 'barh', stacked = True)

unames = ['user_id', 'gender', 'age', 'occupation', 'zip']
users = pd.read_table('data/ch02/movielens/users.dat', sep='::', header=None, names = unames)

rnames = ['user_id', 'movie_id', 'rating', 'timestamp']
ratings = pd.read_table('data/ch02/movielens/ratings.dat', sep='::', header=None, names = rnames)

mnames = ['movie_id', 'title', 'genres']
movies = pd.read_table('data/ch02/movielens/movies.dat', sep='::', header=None, names = mnames)

movies[:5]

data = pd.merge(pd.merge(ratings, users), movies)
data[:5]


# Pivot_table PRODUCES  another DataFrame
mean_ratings = data.pivot_table('rating', index = 'title', columns = 'gender', aggfunc = 'mean')
mean_ratings[:5]

ratings_by_title = data.groupby('title').size()
active_titles = ratings_by_title.index[ratings_by_title >= 250]
active_titles

## use indices to filter down the rows:
mean_ratings = mean_ratings.ix[active_titles]

top_female_ratings = mean_ratings.sort_index(by = 'F', ascending = False)
mean_ratings['diff'] = mean_ratings['M'] - mean_ratings['F'] 
sorted_by_diff = mean_ratings.sort_index(by='diff')

# reversing the order
sorted_by_diff[::-1][:15]

# standard deviation grouped by title:
rating_std_by_title = data.groupby('title')['rating'].std()

# filter down to active_titles
rating_std_by_title = rating_std_by_title.ix[active_titles]
# sort in descending order:
rating_std_by_title.order(ascending=False)[:10]

############################################
##### US names
names1880 = pd.read_csv('data/ch02/names/yob1880.txt', names=['name', 'sex', 'births'])
names1880[:15]

## summing a column grouped by other col
names1880.groupby('sex')['births'].sum()

#### stacking datasets from the same folder (page 30):
years = range(1880, 2011)

pieces = []
columns = ['names', 'sex', 'births', 'year']

for year in years:
    path = 'data/ch02/names/yob%d.txt' % year
    frame = pd.read_csv(path, names = columns)
    
    frame['year'] = year
    pieces.append(frame)

# Concatenate into a single DataFrame
names = pd.concat(pieces, ignore_index = True)

total_births = names.pivot_table('births', index = 'year', columns = 'sex', aggfunc = 'sum')
# showing the end of the range@
total_births.tail()

total_births.plot(title = 'Total births by sex and year')



# Proportion of names born within year:
def add_prop(group):
    # Integer division floors:
    births = group.births #.astype(float)  # no need to cast either denominator or numerator to floating point to compute the fraction
    group['prop'] = births/births.sum()
    return group
names = names.groupby(['year','sex']).apply(add_prop)

names.head()

# to check that all props within groups sum up to 1:
np.allclose(names.groupby(['year', 'sex']).prop.sum(), 1)

# page 33
def get_top1000(group):
    return group.sort_values(by = 'births', ascending = False)[:1000] # using sort_values instead of sort_index here
grouped = names.groupby(['year', 'sex'])
top1000 = grouped.apply(get_top1000)
top1000.index = np.arange(len(top1000))


# dyi approach:
pieces = []
for year, group in names.groupby(['year','sex']):
    pieces.append(group.sort_values(by='births', ascending = False)[:1000])
top1000 = pd.concat(pieces, ignore_index = True)


###
boys = top1000[top1000.sex== 'M']
girls = top1000[top1000.sex== 'F']

total_births = top1000.pivot_table('births', index = 'year', columns = 'names', aggfunc = 'sum')

total_births.info()

subset = total_births[['John', 'Harry', 'Mary', 'Marilyn']]
subset.plot(subplots = True, figsize = (12,10), grid = False, title = 'Number of births per year')

table = top1000.pivot_table('prop', index = 'year', columns = 'sex', aggfunc = sum)
table.plot(title = 'Sum of table1000.prop by year and sex', 
    yticks = np.linspace(0, 1.2, 13), xticks = range(1880, 2020, 10))

df = boys[boys.year == 2010]
## number of distinct names in top 50% births for boys in 2010:
prop_cumsum = df.sort_values(by = 'prop', ascending = False).prop.cumsum()
prop_cumsum.values.searchsorted(0.5)

def get_quantile_count(group, q = 0.5):
    group = group.sort_values(by = 'prop', ascending = False)
    return group.prop.cumsum().values.searchsorted(q)+1

diversity = top1000.groupby(['year', 'sex']).apply(get_quantile_count)
diversity = diversity.unstack('sex')
diversity.head()

diversity.plot(title='Number of popular names in top 50%')

# extract last letter from name column
get_last_letter = lambda x: x[-1]
last_letters = names.names.map(get_last_letter)
last_letters.name = 'last_letter'

table = names.pivot_table('births', index = last_letters, columns = ['sex', 'year'], aggfunc = sum)

# select few rows from pivot table
subtable = table.reindex(columns = [1910, 1960, 2010], level = 'year')

## computing proportion from pivot:
letter_prop = subtable / subtable.sum().astype(float)

# bar plot from pivot:
import matplotlib.pyplot as plt
fig, axes = plt.subplots(2, 1, figsize=(10,8))
letter_prop['M'].plot(kind= 'bar', rot = 0, ax = axes[0], title = 'Male')
letter_prop['F'].plot(kind= 'bar', rot = 0, ax = axes[1], title = 'Female', legend = False)

letter_prop = table/table.sum().astype(float)
# create a time series from pivot table:
dny_ts = letter_prop.ix[['d', 'n', 'y'], 'M'].T
dny_ts.plot()

all_names = top1000.names.unique()
mask = np.array(['lesl' in x.lower() for x in all_names]) 
lesley_like = all_names[mask]

filtered = top1000[top1000.names.isin(lesley_like)]
filtered.groupby('names').births.sum()

table = filtered.pivot_table('births', index='year', columns = 'sex', aggfunc = 'sum')
table.tail()

### plot data split by gender from pivot table
table.plot(style={'M': 'k-', 'F': 'k--'})
