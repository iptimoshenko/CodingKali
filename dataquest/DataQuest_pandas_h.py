import pandas as pd
import numpy as np
from pandas import Series
all_ages = pd.read_csv("https://raw.githubusercontent.com/fivethirtyeight/data/master/college-majors/all-ages.csv")
all_ages.head()

recent_grads = pd.read_csv("https://raw.githubusercontent.com/fivethirtyeight/data/master/college-majors/recent-grads.csv")

rg_cat_counts = recent_grads['Total'].groupby(recent_grads['Major_category']).sum()
aa_cat_counts = all_ages['Total'].groupby(all_ages['Major_category']).sum()

low_wage_proportion = recent_grads['Low_wage_jobs'].sum()/recent_grads['Total'].sum()

majors = list(all_ages['Major_category'].unique())

rg_lower_count = 0
for major in majors:
    rg_lower = all_ages['Unemployment_rate'][all_ages['Major_category'] == major].sum() > \
               recent_grads['Unemployment_rate'][recent_grads['Major_category'] == major].sum()
    #print('all ages unemployment is higher than recent grads in ', major, rg_lower)
    rg_lower_count += rg_lower
print(rg_lower_count)


fandango = pd.read_csv("https://raw.githubusercontent.com/fivethirtyeight/data/master/fandango/fandango_score_comparison.csv")
series_film = fandango['FILM']
print(series_film[:5])
series_rt = fandango['RottenTomatoes']

film_names = series_film.values
rt_scores = series_rt.values
series_custom = Series(rt_scores, index = film_names)
series_custom[['Minions (2015)', 'Leviathan (2014)']]
fiveten = series_custom[5:11]
sorted_index = sorted(series_custom.index.tolist())
series_custom = series_custom.reindex(sorted_index)

# sort series by index
series_custom.sort_index()
# sort series by values
series_custom.sort_values()
## Add 2 series
np.add(series_custom, series_custom)
np.sin(series_custom)
np.max(series_custom)
series_normalized = series_custom/20
criteria_one = series_custom > 50
criteria_two = series_custom < 75
both_criteria = series_custom[criteria_one & criteria_two]

rt_critics = Series(fandango['RottenTomatoes'].values, index=fandango['FILM'])
rt_users = Series(fandango['RottenTomatoes_User'].values, index=fandango['FILM'])
rt_mean = (rt_critics + rt_users)/2
# selecting 2 rows
fandango.iloc[[0, 146]]
fandango.loc[10:13]
fandango.loc[10:13, 'FILM']
fandango.shape

# The set_index() method has a few parameters that allow us to tweak this behavior:
#inplace: If set to True, this parameter will set the index for the current, "live" dataframe, instead of returning a new dataframe.
#drop: If set to False, this parameter will keep the column we specified as the index, instead of dropping it.

fandango_films = fandango.set_index(fandango['FILM'])
print(fandango_films.index)
# When we select multiple rows, pandas returns a dataframe. When we select an individual row, however, it returns a Series object instead.
best_movies_ever = fandango_films.loc[["The Lazarus Effect (2015)", "Gett: The Trial of Viviane Amsalem (2015)", "Mr. Holmes (2015)"]]

# The apply() method in pandas allows us to specify Python logic that we want to evaluate over the Series objects in a dataframe.
# returns the data types as a Series
types = fandango_films.dtypes
# filter data types to just floats, index attributes returns just column names
float_columns = types[types.values == 'float64'].index
# use bracket notation to filter columns to just float columns
float_df = fandango_films[float_columns]

# `x` is a Series object representing a column
deviations = float_df.apply(lambda x: np.std(x))
print(deviations)

double_df = float_df.apply(lambda x: x*2)
print(double_df.head(1))

halved_df = float_df.apply(lambda x: x/2)

rt_mt_user = float_df[['RT_user_norm', 'Metacritic_user_nom']]
rt_mt_deviations = rt_mt_user.apply(lambda x: np.std(x), axis=1)
print(rt_mt_deviations[0:5])

rt_mt_means = rt_mt_user.apply(lambda x: np.mean(x), axis=1)
print(rt_mt_means[0:5])

# Guided Project: Analyzing Thanksgiving
data = pd.read_csv("https://raw.githubusercontent.com/fivethirtyeight/data/master/thanksgiving-2015/thanksgiving-2015-poll-data.csv",
                           encoding="Latin-1")
print(data.columns)
# pandas frequency table:
data["Do you celebrate Thanksgiving?"].value_counts()