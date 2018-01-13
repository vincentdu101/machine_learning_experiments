# Multiple Linear Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

# encode the states column to convert text to number values
# Encoding state data
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()

# we fitted the first column of matrix X and encodes it and reassigns it 
# end result is that the first column values of X is replaced with an encoded 
# int value to show which texts are duplicated
X[:, 3] = labelencoder_X.fit_transform(X[:, 3])

# going to then create dummy encoding where each text value will have its own 
# column of 1 or 0, which is true/false
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()

# Avoiding the Dummy Variable Trap
# X = X[:, 1:]

# splitting the dataset into the training set and test set 
# training set is the set the model will use to learn and develop itself on
# test set is to test that model that is made
from sklearn.cross_validation import train_test_split

# breaking out into training and test sets, test size is what percentage is 
# the size of the test set
X_train, X_test, y_train, y_test = train_test_split(
                                    X, y, test_size = 0.2, random_state = 0 
                                    )

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

# Multiple Linear Regression
# one dependent variable y, but multiple independent variables that can cause 
# or influence the value of the dependent variable

# Assumptions of a linear regression 
# linearity
# homoscedasticity
# multivariate normality
# independence of errors
# lack of multicollinearity

# y = b0 + b1*x1 + b2*x2 + b3*x3 + bn*xn
# going to include dummy variables for state info, will not include all dummy
# variables in one model to avoid dummy variable trap
# always omit one dummy variable

# Building a model
# multiple columns of variables that determine the value of the model 
# have to consider throwing out a columns, why?
## need to filter out garbage columns that dont help predicting the model 
## need to explain model, lowering the number of variables is easier to explain

# 5 methods of building models 
## all-in
### - throw in all variables 
### - do it when you have prior knowledge and you know the columns do determine
### the value 
### - you have to, framework requires variables for example
### - if youre preparing for backward elimination

## backward elimination - step wise progression
### Step 1: select a significance level to stay in the model (ex: 0.05)
### Step 2: fit the full model with all possible predictors
### Step 3: Consider the predictor with the highest P-value: If P > SL go to step 4
### otherwise go to FIN
### Step 4: Remove the predictor, affects other variables
### Step 5: Fit model without this variable, affects other variables
### Then go back to step 3 and keep doing that until you get the to the point 
### where the highest P-value is lower than SL
### When you are done you go to FIN, where the model is ready

## forward selection - step wise progression
### Step 1: select a significance level to enter the model (eg SL = 0.05)
### Step 2: Fit all simple regression models y ~ xn, select the one with the lowest
### P-value
### Step 3: Keep this variable and fit all possible models with one extra predictor
### added to the one you already have (add in each variable separately)
### Step 4: Consider the predictor with the lowest P-value, if P < SL go to step 3
### otherwise go to FIN
### Only stop when we have a variable P value that is greater than SL 
### At that point we go to FIN, where the model is ready 

## bidirectional elimination - step wise progression
### Step 1: select a significance level to enter and to stay in the model, eg
### SLENTER = 0.05, SLSTAY = 0.05
### Step 2: Perform the next step of forward selection (new variables must have 
### P < SLENTER to enter)
### Step 3: Perform all steps of backward elimination (old variables must have 
### P < SLSTAY to stay)
### Step 4: No new variables can enter and no new variables can stay

## score comparison - all possible models
### Step 1: select a criterion of goodness of fit (Akaike criterion)
### Step 2: Construct all possible regression models 2^n - 1 total combinations
### Step 3: Select the one with the best criterion
### FIN: Your model is ready
### very resource consuming approach 

# dependent variable is profit, independent variables are other columns except index 

# Fitting Multiple lInear regression to the training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)























































