from Classes_tutorial import Dataset


# Sets
unique_animals = set(["Dog", "Cat", "Hippo", "Dog", "Cat", "Dog", "Dog", "Cat"])
print(unique_animals)
unique_animals.add("Tiger")
unique_animals.remove("Dog")
list(unique_animals)

gender = []
for row in legislators:
    gender.append(row[3])
gender = set(gender)
print(gender)


party = []
for line in legislators:
    party.append(line[6])
party = set(party)
print(party)

for line in legislators:
    print(line)

birth_years = []

for line in legislators:
    date = line[2].split("-")
    year = date[0]
    birth_years.append(year)

birth_months = []
for line in legislators:
    date = line[2].split("-")
    if len(date)>1:
        month = int(date[1])
        birth_months.append(month)

from itertools import groupby
[len(list(group)) for key, group in groupby(birth_months)]


## Error handling:
converted_years = []
for item in birth_years:
    try:
        year = int(item)
        converted_years.append(year)
    except:
        pass


birth_years = []

for line in legislators:
    date = line[2].split("-")
    try:
        birth_year = int(date[0])
    except:
        birth_year=0
    line.append(birth_year)

# List comprehensions:
# Append a column
ships = ["Andrea Doria", "Titanic", "Lusitania"]
cars = ["Ford Edsel", "Ford Pinto", "Yugo"]

for i, value in enumerate(ships):
    print(value)
    print(cars[i])