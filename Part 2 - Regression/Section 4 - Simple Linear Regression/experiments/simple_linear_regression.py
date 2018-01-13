# Simple Linear Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

# splitting the dataset into the training set and test set 
# training set is the set the model will use to learn and develop itself on
# test set is to test that model that is made

# breaking out into training and test sets, test size is what percentage is 
# the size of the test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

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

# best fitting line - simple linear regression
# y = b0 + b1 * x 
# y is the dependent variable and its value is dependent on the x variable 
# which is the independent variable
# b0 is the constant when x is 0, the initial default coefficient that crosses
# the y-axis, the base value
# b1 is the incremental change in value with x changing positively or negatively

# how do we determine the best fitting line
# ordinary least squares 
# draw different lines of sum of difference between observered and modeled values 
# squared, then finds the smallest sum

# linear regression model learned the correlations between experience and salary
# and determined a model to reflect both variables
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# predicting the test set results
# vector of predicted salaries for all values of test set
y_pred = regressor.predict(X_test)

# visualize the training set results
plt.scatter(X_train, y_train, color='red')
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title("Salary vs Experience (Training Set)")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.show()

# visualize the testing set results
plt.scatter(X_test, y_test, color='red')
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title("Salary vs Experience (Testing Set)")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.show()
 








































