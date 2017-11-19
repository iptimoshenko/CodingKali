import numpy as np
world_alcohol = np.genfromtxt("world_alcohol.csv", delimiter=",", dtype = "U75", skip_header=True)
print(world_alcohol)

uruguay_other_1986 =  world_alcohol[1][4]
third_country = world_alcohol[2][2]

alcohol_consumption = world_alcohol[:, 4]
countries = world_alcohol[:, 2]

world_alcohol[1:3, :]
world_alcohol[1:3, 0:2]

first_twenty_regions = world_alcohol[:20,1:3]
countries = world_alcohol[:, 2]
