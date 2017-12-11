import os
os.chdir('C:/Users/Asus/PycharmProjects/CodingKali/dataquest')
import pandas as pd
import matplotlib.pyplot as plt

women_degrees = pd.read_csv('percent-bachelors-degrees-women-usa.csv')
plt.plot(women_degrees["Year"], women_degrees["Biology"])
women_degrees["men_biology"] = 100 - women_degrees["Biology"]

fig, ax = plt.subplots()
ax.plot(women_degrees["Year"], women_degrees["Biology"], c="blue", label="Women")
ax.plot(women_degrees["Year"], women_degrees["men_biology"], c="green", label="Men")
ax.legend(loc="upper right")
ax.tick_params(bottom="off", top="off", left="off", right="off")
plt.title("Percentage of Biology Degrees Awarded by Gender")
ax.spines["right"].set_visible(False)
ax.spines["left"].set_visible(False)
ax.spines["bottom"].set_visible(False)
ax.spines["top"].set_visible(False)
plt.show()


major_cats = ['Biology', 'Computer Science', 'Engineering', 'Math and Statistics']
cb_dark_blue = (0/255,107/255,164/255)
cb_orange = (255/255,128/255,14/255)

fig = plt.figure(figsize=(12, 12))
for sp in range(0,4):
    ax = fig.add_subplot(2,2,sp+1)
    ax.plot(women_degrees['Year'], women_degrees[major_cats[sp]], c=cb_dark_blue, label='Women', linewidth=3)
    ax.plot(women_degrees['Year'], 100-women_degrees[major_cats[sp]], c=cb_orange, label='Men', linewidth=3)
    ax.set_xlim(1968,2011)
    ax.set_ylim(0,100)
    ax.tick_params(bottom="off", top="off", left="off", right="off")
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["top"].set_visible(False)
    plt.title(major_cats[sp])

# Calling pyplot.legend() here will add the legend to the last subplot that was created.
plt.legend(loc='upper right')
plt.show()


## 1 line 6 charts
stem_cats = ['Engineering', 'Computer Science', 'Psychology', 'Biology', 'Physical Sciences', 'Math and Statistics']

fig = plt.figure(figsize=(18, 3))

for sp in range(0,6):
    ax = fig.add_subplot(1,6,sp+1)
    ax.plot(women_degrees['Year'], women_degrees[stem_cats[sp]], c=cb_dark_blue, label='Women', linewidth=3)
    ax.plot(women_degrees['Year'], 100-women_degrees[stem_cats[sp]], c=cb_orange, label='Men', linewidth=3)
    for key,spine in ax.spines.items():
        spine.set_visible(False)
    ax.set_xlim(1968, 2011)
    ax.set_ylim(0,100)
    ax.set_title(stem_cats[sp])
    ax.tick_params(bottom="off", top="off", left="off", right="off")

plt.legend(loc='upper right')
plt.show()


# remove legend, annotate lines
fig = plt.figure(figsize=(18, 3))

for sp in range(0,6):
    ax = fig.add_subplot(1,6,sp+1)
    ax.plot(women_degrees['Year'], women_degrees[stem_cats[sp]], c=cb_dark_blue, label='Women', linewidth=3)
    ax.plot(women_degrees['Year'], 100-women_degrees[stem_cats[sp]], c=cb_orange, label='Men', linewidth=3)
    for key,spine in ax.spines.items():
        spine.set_visible(False)
    ax.set_xlim(1968, 2011)
    ax.set_ylim(0,100)
    if sp==0:
        ax.text(2005, 87, "Men")
        ax.text(2002, 8, "Men")
    if sp==5:
        ax.text(2005, 62, "Men")
        ax.text(2001, 35, "Men")
    ax.set_title(stem_cats[sp])
    ax.tick_params(bottom="off", top="off", left="off", right="off")
plt.legend(loc='upper right')
plt.show()

# 6*3 grid
cb_dark_blue = (0/255,107/255,164/255)
cb_orange = (255/255, 128/255, 14/255)
stem_cats = ['Psychology', 'Biology', 'Math and Statistics', 'Physical Sciences', 'Computer Science', 'Engineering', 'Computer Science']
lib_arts_cats = ['Foreign Languages', 'English', 'Communications and Journalism', 'Art and Performance', 'Social Sciences and History']
other_cats = ['Health Professions', 'Public Administration', 'Education', 'Agriculture','Business', 'Architecture']

fig = plt.figure(figsize=(21, 9))

for sp in range(0,6):
    ax = fig.add_subplot(6,3,(3*sp+1))
    ax.plot(women_degrees['Year'], women_degrees[stem_cats[sp]], c=cb_dark_blue, label='Women', linewidth=3)
    ax.plot(women_degrees['Year'], 100-women_degrees[stem_cats[sp]], c=cb_orange, label='Men', linewidth=3)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.set_xlim(1968, 2011)
    ax.set_ylim(0,100)
    ax.axhline(50, c=(171 / 255, 171 / 255, 171 / 255), alpha=0.3)  # alpha determines transparency
    ax.set_title(stem_cats[sp])
    ax.set_yticks([0, 100])
    ax.tick_params(bottom="off", top="off", left="off", right="off", labelbottom='off')
    if sp == 0:
        ax.text(2005, 87, 'Men')
        ax.text(2002, 8, 'Women')
    elif sp == 5:
        ax.text(2005, 62, 'Men')
        ax.text(2001, 35, 'Women')
        ax.tick_params(labelbottom='on')

for sp in range(0,5):
    ax = fig.add_subplot(6,3, (3*sp+2))
    ax.plot(women_degrees['Year'], women_degrees[lib_arts_cats[sp]], c=cb_dark_blue, label='Women', linewidth=3)
    ax.plot(women_degrees['Year'], 100-women_degrees[lib_arts_cats[sp]], c=cb_orange, label='Men', linewidth=3)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.set_xlim(1968, 2011)
    ax.set_ylim(0,100)
    ax.set_yticks([0, 100])
    ax.axhline(50, c=(171 / 255, 171 / 255, 171 / 255), alpha=0.3)  # alpha determines transparency
    ax.set_title(lib_arts_cats[sp])
    ax.tick_params(bottom="off", top="off", left="off", right="off", labelbottom='off')
    if sp == 0:
        ax.text(2005, 87, 'Men')
        ax.text(2002, 8, 'Women')
    elif sp == 4:
        ax.tick_params(labelbottom='on')

for sp in range(0,6):
    ax = fig.add_subplot(6,3, (3*sp+3))
    ax.plot(women_degrees['Year'], women_degrees[other_cats[sp]], c=cb_dark_blue, label='Women', linewidth=3)
    ax.plot(women_degrees['Year'], 100-women_degrees[other_cats[sp]], c=cb_orange, label='Men', linewidth=3)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.set_xlim(1968, 2011)
    ax.set_ylim(0,100)
    ax.set_yticks([0, 100])
    ax.set_title(other_cats[sp])
    ax.axhline(50, c=(171/255, 171/255, 171/255), alpha=0.3) # alpha determines transparency
    ax.tick_params(bottom="off", top="off", left="off", right="off", labelbottom='off')
    if sp == 0:
        ax.text(2007, 90, 'Men')
        ax.text(2006, 5, 'Women')
    elif sp == 5:
        ax.text(2007, 60, 'Men')
        ax.text(2006, 18, 'Women')
        ax.tick_params(labelbottom='on')
plt.savefig("Gender_comparison_all_degrees.png")
plt.show()
