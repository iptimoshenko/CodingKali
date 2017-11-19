import pandas as pd
import numpy as np
from pandas import Series

recent_grads = pd.read_csv("https://raw.githubusercontent.com/fivethirtyeight/data/master/college-majors/recent-grads.csv")
recent_grads.tail()
recent_grads.describe()
raw_data_count = recent_grads.shape[0]
recent_grads = recent_grads.dropna(axis=0)
recent_grads.shape[0]

recent_grads.plot(x='Sample_size', y='Employed', kind='scatter', title="Employed vs Sample_size", figsize=(5,10))