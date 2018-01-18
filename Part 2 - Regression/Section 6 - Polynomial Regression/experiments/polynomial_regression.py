# Polynomial Regression

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

# polynomial regression description
# similar to multiple linear regression but different powers 
# y = bo + b1x1 + b2x^2 + b3x^3 + ... bnx^n
# parabolic factor that helps better align with growing different rates of change
# of the dependent variable values
# ex: how pandemic spreads among populations over time
# special case of multiple linear regression

# why linear still?
# x variables is not important, it is about the coefficients themselves where constant
# and combination of coefficients equate to the dependent variable
# non-linear equation cannot replace coefficients with other coefficients to 
# create linear equation
# not a linear regressor anymore in other videos- why?

# we will use all of the dataset to make model rather than have a test set to allow
# maximum info used to determine model

# pure linear regression model to compare results between linear and polynomial 
# regression models
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)

# transforms features of independent variables into polynomial independent variables
# a higher degree of freedom increases the accuracy of the model
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X)

# second linear regression model takes into account the newly transformed polynomial
# independent variables
from sklearn.linear_model import LinearRegression
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)

# visualizing the linear regression results
plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg.predict(X), color = 'blue')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# visualizing the polynomial regression results
plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg_2.predict(poly_reg.fit_transform(X)), color = 'blue')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()













































