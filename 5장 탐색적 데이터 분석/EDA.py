import pandas as pd
import numpy as np
import os

os.getcwd()

fueleco = pd.read_csv("pandas_cookbook\datas\cars.csv")
fueleco

fueleco.mean()

fueleco.std()
