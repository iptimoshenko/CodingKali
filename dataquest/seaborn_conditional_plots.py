import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

titanic = pd.read_csv("train.csv")
titanic = titanic.loc[:, ["Survived", "Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]]
titanic = titanic.dropna()


sns.distplot(titanic["Fare"])
sns.distplot(titanic["Age"])
plt.show()

# various background styles
sns.set_style("white")
sns.kdeplot(titanic["Age"], shade=True)
plt.xlabel("Age")
sns.despine(left=True, bottom=True)

# Condition on unique values of the "Survived" column.
g = sns.FacetGrid(titanic, col="Pclass", size=6) # size specifies the height in inches for each plot
# For each subset of values, generate a kernel density plot of the "Age" columns.
g.map(sns.kdeplot, "Age", shade=True)
sns.despine(left=True, bottom=True)
plt.show()

# Facet grid based on 3 conditions
g = sns.FacetGrid(titanic, col="Survived", row="Pclass", hue="Sex", size=3)
g.map(sns.kdeplot, "Age", shade=True).add_legend()
sns.despine(left=True, bottom=True)
plt.show()

