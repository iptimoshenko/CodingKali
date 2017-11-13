import pandas as pd
food_info = pd.read_csv('food_info.csv')
col_names = food_info.columns.tolist()
print(col_names)
print(food_info.loc[:3])

food_info['Normalized_Protein'] = food_info["Protein_(g)"]/food_info["Protein_(g)"].max()
food_info['Normalized_Fat'] = food_info["Lipid_Tot_(g)"]/food_info["Lipid_Tot_(g)"].max()
food_info["Norm_Nutr_Index"] = 2*food_info["Normalized_Protein"] - 0.75*food_info["Normalized_Fat"]
# Sorts the DataFrame in-place, rather than returning a new DataFrame.
food_info.sort_values("Norm_Nutr_Index", inplace=True, ascending=False)

titanic_survival = pd.read_csv("titanic_survival.csv")
age = titanic_survival["age"]
print(age.loc[10:20])
age_is_null = pd.isnull(age)
age_null_true = age[age_is_null]
age_null_count = len(age_null_true)
print(age_null_count)


age_is_null = pd.isnull(titanic_survival["age"])
good_age = titanic_survival["age"][age_is_null==False]
correct_mean_age = sum(good_age) / len(good_age)

passenger_survival = titanic_survival.pivot_table(index="pclass", values="survived")
passenger_age = titanic_survival.pivot_table(index="pclass", values="age", aggfunc = numpy.mean)

import numpy as np
port_stats = titanic_survival.pivot_table(index="embarked", values=["fare", "survived"], aggfunc = np.sum)
print(port_stats)

drop_na_rows = titanic_survival.dropna(axis=0) # axis=1 will drop columns with missing values
#how : {‘any’, ‘all’}
# any : if any NA values are present, drop that label
# all : if all values are NA, drop that label
drop_na_columns = titanic_survival.dropna(axis=1, how='any')
# subset : array-like
# Labels along other axis to consider, e.g. if you are dropping rows these would be a list of columns to include
new_titanic_survival = titanic_survival.dropna(axis=0, subset=["age", "sex"])

#iloc[] stands for integer location# We have already sorted new_titanic_survival by age
first_five_rows = new_titanic_survival.iloc[0:5]
first_ten_rows = new_titanic_survival.iloc[0:10]
row_position_fifth = new_titanic_survival.iloc[4]
# these  work just like column labels, and can be values like numbers, characters, and strings
row_index_25 = new_titanic_survival.loc[25]

# We can also index columns using both the loc[] and iloc[] methods.
# With .loc[], we specify the column label strings as we have in the earlier exercises
# in this missions. With iloc[], we simply use the integer number of the column,
# starting from the left-most column which is 0. Similar to indexing with NumPy arrays,
# you separate the row and columns with a comma, and can use a colon to specify a range or as a wildcard.
first_row_first_column = new_titanic_survival.iloc[0,0]
all_rows_first_three_columns = new_titanic_survival.iloc[:,0:3]
row_index_83_age = new_titanic_survival.loc[83,"age"]
row_index_766_pclass = new_titanic_survival.loc[766,"pclass"]

row_index_1100_age = new_titanic_survival.loc[1100, "age"]
row_index_25_survived = new_titanic_survival.loc[25, "survived"]
five_rows_three_cols = new_titanic_survival.iloc[:5, :3]
titanic_reindexed = new_titanic_survival.reset_index(drop = True)
print(titanic_reindexed.iloc[:5, :3])

# By default, DataFrame.apply() will iterate through each column in a DataFrame,
# and perform on each column. When we create our function, we give it one parameter,
# apply() method passes each column to the parameter as a pandas series.
def num_null(column):
    column_null = pd.isnull(column)
    null = column[column_null]
    return len(null)
column_null_count = titanic_survival.apply(num_null)

# to iterate .apply over rows, use argument axis=1
def is_minor(row):
    age = row["age"]
    if pd.isnull(age):
        return "unknown"
    elif age < 18:
        return 'minor'
    else:
        return 'adult'

age_labels = titanic_survival.apply(is_minor, axis=1)
age_group_survival = titanic_survival.pivot_table(index="age_labels", values="survived", aggfunc = np.mean)
