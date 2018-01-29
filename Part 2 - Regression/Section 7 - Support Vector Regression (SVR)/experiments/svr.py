# Support Vector Regression (SVR)

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# feature scaling - normalize values so they are in the same range for measurement
# easier to have models analyze differences and not skew results 
# for svr we need to use it because this svr class does not have include feature
# scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y)

# we do not need to do feature scaling to the y variable column

# fitting the new regression model to the dataset
from sklearn.svm import SVR
regressor = SVR(kernel = "rbf").fit(X, y)

# predicting a new result with svr
# need to transform it since the current regressor is using x and y coordinates
y_pred = regressor.predict(6.5)
y_pred = sc_y.inverse_transform(y_pred)

# visualizing the svr results
# you will see that the CEO salary is an outlier, since its too far from the 
# other points the line does not follow it completely
plt.scatter(X, y, color = 'red')
plt.plot(X, regressor.predict(X), color = 'blue')
plt.title('Truth or Bluff (SVR Model)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()









































