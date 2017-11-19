import pandas as pd
import numpy as np
from pandas import Series
import matplotlib.pyplot as plt
from numpy import arange

reviews = pd.read_csv("https://raw.githubusercontent.com/fivethirtyeight/data/master/fandango/fandango_score_comparison.csv")
reviews.head()
norm_reviews = reviews[["FILM", "RT_user_norm", "Metacritic_user_nom", "IMDB_norm", "Fandango_Ratingvalue", "Fandango_Stars"]]
num_cols = ['RT_user_norm', 'Metacritic_user_nom', 'IMDB_norm', 'Fandango_Ratingvalue', 'Fandango_Stars']

bar_heights = norm_reviews[num_cols].iloc[0].values
bar_positions = arange(5) + 0.75
tick_positions = range(1,6)
num_cols = ['RT_user_norm', 'Metacritic_user_nom', 'IMDB_norm', 'Fandango_Ratingvalue', 'Fandango_Stars']

fig, ax = plt.subplots()
ax.bar(bar_positions, bar_heights, 0.5)
ax.set_xticks(tick_positions)
ax.set_xticklabels(num_cols, rotation = 90)
ax.set_xlabel("Rating Source")
ax.set_ylabel("Average Rating")
ax.set_title("Average User Rating For Avengers: Age of Ultron (2015)")
plt.show()

# horizontal bar chart
fig, ax = plt.subplots()
bar_widths = norm_reviews[num_cols].iloc[0].values
bar_positions = arange(5) + 0.75
ax.barh(bar_positions, bar_widths, 0.5)
tick_positions = arange(5) + 1
ax.set_yticks(tick_positions)
ax.set_yticklabels(num_cols)
ax.set_xlabel("Average Rating")
ax.set_ylabel("Rating Source")
ax.set_title("Average User Rating For Avengers: Age of Ultron (2015)")
plt.show()


# scatter plot
fig, ax = plt.subplots()
ax.scatter(norm_reviews["Fandango_Ratingvalue"], norm_reviews["RT_user_norm"])
ax.set_xlabel("Fandango")
ax.set_ylabel("Rotten Tomatoes")
plt.show()

fig = plt.figure(figsize=(5,10))
ax1 = fig.add_subplot(2,1,1)
ax2 = fig.add_subplot(2,1,2)
ax1.scatter(norm_reviews["Fandango_Ratingvalue"], norm_reviews["RT_user_norm"])
ax1.set_xlabel("Fandango")
ax1.set_ylabel("Rotten Tomatoes")
ax2.scatter(norm_reviews["RT_user_norm"], norm_reviews["Fandango_Ratingvalue"])
ax1.set_xlabel("Rotten Tomatoes")
ax1.set_ylabel("Fandango")
plt.show()

y_values= ["RT_user_norm", "Metacritic_user_nom", "IMDB_norm"]
fig = plt.figure(figsize=(5,10))
ax1 = fig.add_subplot(3,1,1)
ax2 = fig.add_subplot(3,1,2)
ax3 = fig.add_subplot(3,1,3)
ax1.scatter(norm_reviews["Fandango_Ratingvalue"], norm_reviews["RT_user_norm"])
ax1.set_xlabel("Fandango")
ax1.set_ylabel("Rotten Tomatoes")
ax1.set_xlim(0,5)
ax1.set_ylim(0,5)
ax2.scatter(norm_reviews["Fandango_Ratingvalue"], norm_reviews["Metacritic_user_nom"])
ax2.set_xlabel("Fandango")
ax2.set_ylabel("Metacritic")
ax2.set_xlim(0,5)
ax2.set_ylim(0,5)
ax3.scatter(norm_reviews["Fandango_Ratingvalue"], norm_reviews["IMDB_norm"])
ax3.set_xlabel("Fandango")
ax3.set_ylabel("IMDB")
ax3.set_xlim(0,5)
ax3.set_ylim(0,5)
plt.show()


fandango_distribution = norm_reviews["Fandango_Ratingvalue"].value_counts().sort_index()
# to sort frequency table by value, not frequency
sorted_freq_counts = fandango_distribution.sort_index()

fandango_distribution = norm_reviews["Fandango_Ratingvalue"].value_counts().sort_index()
imdb_distribution = norm_reviews["IMDB_norm"].value_counts().sort_index()

print(fandango_distribution)
print(imdb_distribution)


########################
### Histograms #########
########################
# By default, matplotlib will:
# calculate the minimum and maximum value from the sequence of values we passed in
# create 10 bins

fig, ax = plt.subplots()
ax.hist(norm_reviews["Fandango_Ratingvalue"], range=(0,5))
plt.show()

fig = plt.figure(figsize=(5,20))
ax1 = fig.add_subplot(4,1,1)
ax2 = fig.add_subplot(4,1,2)
ax3 = fig.add_subplot(4,1,3)
ax4 = fig.add_subplot(4,1,4)
ax1.hist(norm_reviews["Fandango_Ratingvalue"], bins=20, range=(0,5))
ax1.set_ylim(0, 50)
ax1.set_ylabel("Fandango")
ax2.hist(norm_reviews["RT_user_norm"], bins=20, range=(0,5))
ax2.set_ylim(0, 50)
ax2.set_ylabel("Rotten tomatoes")
ax3.hist(norm_reviews["Metacritic_user_nom"], bins=20, range=(0,5))
ax3.set_ylim(0, 50)
ax3.set_ylabel("Metacritic")
ax4.hist(norm_reviews["IMDB_norm"], bins=20, range=(0,5))
ax4.set_ylim(0, 50)
ax4.set_ylabel("IMDB")
plt.show()


####################################
#### Box plots
fix, ax = plt.subplots()
ax.boxplot(norm_reviews["RT_user_norm"])
ax.set_ylim(0, 5)
ax.set_xticklabels(["Rotten Tomatoes"])
plt.show()

num_cols = ['RT_user_norm', 'Metacritic_user_nom', 'IMDB_norm', 'Fandango_Ratingvalue', 'Fandango_Stars']
fix, ax = plt.subplots()
ax.boxplot(norm_reviews[num_cols].values)
ax.set_xticklabels(num_cols, rotation=90)
ax.set_ylim(0,5)
plt.show()