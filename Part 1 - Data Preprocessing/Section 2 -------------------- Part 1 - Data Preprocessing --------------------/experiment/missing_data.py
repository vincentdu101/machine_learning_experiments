# Data Preprocessing

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv("Data.csv")

# independent variable matrix of features
# iloc[all_lines, take all columns but last]
X = dataset.iloc[:, :-1].values

# dependent variable matrix
# iloc[all_lines, take column via index]
Y = dataset.iloc[:, 3].values

# taking care of missing data
from sklearn.preprocessing import Imputer

# look for NaN values as missing data, then using the mean of the columns to 
# implement as the replacement value
imputer = Imputer(missing_values = "NaN", strategy = "mean", axis = 0)

# take columns with index 1 and 2, upper bound in python arrays are ignored
# fit imputer object to the X matrix 
imputer = imputer.fit(X[:, 1:3])

# imputer replaces the missing values in X matrix and reassign X to the X matrix
X[:, 1:3] = imputer.transform(X[:, 1:3])
