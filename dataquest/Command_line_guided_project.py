import pandas as pd
hn_stories = pd.read_csv("stories.csv")


for name, row in domains.items():
    print("{0}: {1}".format(name, row))


# to parse the date: dateutil.parser.parse