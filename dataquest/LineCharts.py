import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

unrate = pd.read_csv("dataquest/data/unrate.csv")
unrate["DATE"] = pd.to_datetime(unrate["DATE"])
print(unrate.iloc[:12])

plt.plot()
plt.show()

# Because we didn't pass in any arguments, the plot() function would generate an empty plot with just the axes and ticks
#  and the show() function would display that plot.
# This is because every time we call a pyplot function, the module maintains and updates the plot internally
# (also known as state).
# When we call show(), the plot is displayed and the internal state is destroyed.

plt.plot(unrate["DATE"].iloc[:12], unrate["UNRATE"].iloc[:12])
plt.xticks(rotation=90)
plt.xlabel("Month")
plt.ylabel("Unemployment rate")
plt.title("Monthly Unemployment Trends, 1948")
plt.show()

# xlabel(): accepts a string value, which gets set as the x-axis label.
# ylabel(): accepts a string value, which is set as the y-axis label.
# title(): accepts a string value, which is set as the plot title.

fig = plt.figure()
# While plots are represented using instances of the Axes class, they're also often referred to as subplots in
# matplotlib.
# Figure.add_subplot. This will return a new Axes object, which needs to be assigned to a variable:

axes_obj = fig.add_subplot(nrows, ncols, plot_number)

ax1 = fig.add_subplot(2, 1, 1)
ax2 = fig.add_subplot(2, 1, 2)
plt.show()

# Each time we want to generate a line chart, we need to call Axes.plot() and pass in the data we want to use in that plot.
# The unit for both width and height values is inches. The dpi parameter, or dots per inch
fig = plt.figure(figsize=(12, 12)) # width by height
ax1 = fig.add_subplot(5, 1, 1)
ax2 = fig.add_subplot(5, 1, 2)
ax3 = fig.add_subplot(5, 1, 3)
ax4 = fig.add_subplot(5, 1, 4)
ax5 = fig.add_subplot(5, 1, 5)
ax1.plot(unrate[0:12]["DATE"], unrate[0:12]["UNRATE"])
ax2.plot(unrate[12:24]["DATE"], unrate[12:24]["UNRATE"])
ax3.plot(unrate[24:36]["DATE"], unrate[24:36]["UNRATE"])
ax4.plot(unrate["DATE"].iloc[36:48], unrate["UNRATE"].iloc[36:48])
ax5.plot(unrate["DATE"].iloc[48:60], unrate["UNRATE"].iloc[48:60])
plt.show()

# pandas.Series.dt to extract month
unrate['MONTH'] = unrate['DATE'].dt.month

fig = plt.figure(figsize=(6,3))
plt.plot(unrate[0:12]["DATE"], unrate[0:12]["UNRATE"], c="green")
plt.plot(unrate[12:24]["DATE"], unrate[12:24]["UNRATE"], c="green")


unrate.rename(columns={'DATE': 'DATE', 'UNRATE': 'VALUE'}, inplace=True)
fig = plt.figure(figsize=(10, 6))
colours = ['red', 'blue', 'green', 'orange', 'black']
for i in range(5):
    start_index = i*12
    end_index = (i+1)*12
    subset = unrate[start_index:end_index]
    label = str(1948 + i)
    plt.plot(subset["MONTH"], subset["VALUE"], c=colours[i], label=label)
plt.title("Monthly unemployemnt trends, 1948 - 1952")
plt.xlabel("Month, integer")
plt.ylabel("Unemployment rate, percent")
plt.legend(loc='upper right')
plt.show()

# When we use plt.plot() and plt.legend(), the Axes.plot() and Axes.legend() methods are called under the hood and
#  parameters passed to the calls. When we need to create a legend for each subplot, we can use Axes.legend() instead.