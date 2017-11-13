import csv
with open('guns.csv', 'r') as f:
    csvreader = csv.reader(f)
    data = list(csvreader)
print(data[:5])

class Dataset:
    def __init__(self):
        self.type = "csv"

dataset = Dataset()
# When creating this object, the Python interpreter uses the special __init__()
#  method we defined to instantiate the object. This creates the new object
# and then sets those attributes to the instance.

print(dataset.type)

# Python uses this self variable to refer to the created object so you can
# interact with the instance data. If you didn't have self, then the class
# wouldn't know where to store the internal data you wanted to keep.
# It's named "self" by convention



f = open('nfl.csv', 'r')
csvreader = csv.reader(f)
nfl_data = list(csvreader)

nfl_dataset = Dataset(nfl_data)
nfl_dataset.print_data(5)


class Dataset:
    def __init__(self, data):
        self.header = data[0]
        self.data = data[1:]

    def __str__(self):
        first_10 = str(self.data[:10])
        return first_10

    def column(self, label):
        if label not in self.header:
            return None

        index = 0
        for idx, element in enumerate(self.header):
            if label == element:
                index = idx

        column = []
        for row in self.data:
            column.append(row[index])
        return column

    def count_unique(self, label):
        unique_results = set(self.column(label))
        count = len(unique_results)
        return count


nfl_dataset = Dataset(nfl_data)
print(nfl_dataset)


nfl_dataset = Dataset(nfl_data)
total_years = nfl_dataset.count_unique('year')


nfl_dataset = Dataset(nfl_data)
year_column = nfl_dataset.column('year')
player_column = nfl_dataset.column('player')

nfl_dataset = Dataset(nfl_data)
nfl_header = nfl_dataset.header

# One special method is __str__() which tells the python interpreter how to represent your object as a string.

