# Regression Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# splitting the dataset into the training set and test set 
# training set is the set the model will use to learn and develop itself on
# test set is to test that model that is made
#from sklearn.cross_validation import train_test_split

# breaking out into training and test sets, test size is what percentage is 
# the size of the test set
#X_train, X_test, y_train, y_test = train_test_split(
#                                    X, y, test_size = 0.2, random_state = 0 
#                                    )

# model can learn from training set whether there is a correlation categories 
# will mean a customer will buy or not buy a product, then test on test set 
# to see if it can predict the values

# feature scaling - normalize values so they are in the same range for measurement
# easier to have models analyze differences and not skew results 
#from sklearn.preprocessing import StandardScaler
#sc_X = StandardScaler()
#X_train = sc_X.fit_transform(X_train)

# do not need to fit_transform, already done on training set, model will need
# to accept original dataset
#X_test = sc_X.transform(X_test)

# we do not need to do feature scaling to the y variable column

# fitting the new regression model to the dataset
# create the regressor

# predicting a new result with regression
pred = regressor.predict(6.5)

# visualizing the regression results
plt.scatter(X, y, color = 'red')
plt.plot(X, regressor.predict(X), color = 'blue')
plt.title('Truth or Bluff (Regression Model)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# visualizing the regression results for higher resolution and smoother curvey
# create smoother line by having more points to plot broken down by incrementals 
# of 0.1
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = np.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff (Regression Model)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()











































