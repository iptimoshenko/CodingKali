import os
os.chdir('C:/Users/Asus/PycharmProjects/CodingKali/dataquest/data')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn
from sklearn.metrics import mean_squared_error

dc_listings = pd.read_csv("listings.csv")
print(dc_listings.head(1))

dc_listings["distance"] = abs(dc_listings["accommodates"]-3)
dc_listings["distance"].value_counts()


np.random.seed(1)
shuffled_index = np.random.permutation(len(dc_listings))
dc_listings = dc_listings.loc[shuffled_index]
dc_listings.sort_values("distance")
print(dc_listings.iloc[0:10]['price'])

# cleaning commas and $ out of price
stripped_price = dc_listings['price'].astype('str').replace(',', '').replace('$', '')
dc_listings['price'] = stripped_price.astype('float')
mean_price = dc_listings.iloc[:5]['price'].mean()
print(mean_price)

def predict_price(new_listing):
    temp_df = dc_listings
    temp_df["distance"] = abs(temp_df["accommodates"]-new_listing)
    temp_df = temp_df.sort_values("distance")
    mean_price = temp_df.iloc[:5]['price'].mean()
    return(mean_price)

acc_one = predict_price(1)
acc_two = predict_price(2)
acc_four = predict_price(4)

## linear functions
seaborn.set(style='darkgrid')

def draw_secant(x_values):
    x = np.linspace(-20, 30, 100)
    y = -1 * (x ** 2) + x * 3 - 1
    plt.plot(x, y)
    y_values = []
    for x in x_values:
        y_values.append(-1 * (x ** 2) + x * 3 - 1)
    m = (y_values[1] - y_values[0]) / (x_values[1] - x_values[0])
    b = y_values[1] - m * x_values[1]
    secant = m * x + b
    plt.plot(x, secant, c="green")
    plt.show()

draw_secant([3, 5])


##########################################################
########## Linear Regression
import matplotlib.pyplot as plt
# For prettier plots.
import seaborn
fig = plt.figure(figsize=(7,15))

ax1 = fig.add_subplot(3, 1, 1)
ax2 = fig.add_subplot(3, 1, 2)
ax3 = fig.add_subplot(3, 1, 3)

train.plot(x="Garage Area", y="SalePrice", ax=ax1, kind="scatter")
train.plot(x="Gr Liv Area", y="SalePrice", ax=ax2, kind="scatter")
train.plot(x="Overall Cond", y="SalePrice", ax=ax3, kind="scatter")

plt.show()

train[['Garage Area', 'Gr Liv Area', 'Overall Cond', 'SalePrice']].corr()
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(train[['Gr Liv Area']], train['SalePrice'])
print(lr.coef_)
print(lr.intercept_)

a0 = lr.intercept_
a1 = lr.coef_

lr = LinearRegression()
lr.fit(train[['Gr Liv Area']], train['SalePrice'])

train_predictions = lr.predict(train[['Gr Liv Area']])
test_predictions = lr.predict(test[['Gr Liv Area']])

train_mse = mean_squared_error(train_predictions, train['SalePrice'])
test_mse = mean_squared_error(test_predictions, test['SalePrice'])

train_rmse = np.sqrt(train_mse)
test_rmse = np.sqrt(test_mse)

print(train_rmse)
print(test_rmse)

cols = ['Overall Cond', 'Gr Liv Area']
lr.fit(train[cols], train['SalePrice'])
train_predictions = lr.predict(train[cols])
test_predictions = lr.predict(test[cols])

train_rmse_2 = np.sqrt(mean_squared_error(train_predictions, train['SalePrice']))
test_rmse_2 = np.sqrt(mean_squared_error(test_predictions, test['SalePrice']))

print(train_rmse_2)
print(test_rmse_2)