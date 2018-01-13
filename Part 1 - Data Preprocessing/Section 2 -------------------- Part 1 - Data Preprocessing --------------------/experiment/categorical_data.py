# Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 3].values

# Taking care of missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])

# Encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()

# we fitted the first column of matrix X and encodes it and reassigns it 
# end result is that the first column values of X is replaced with an encoded 
# int value to show which texts are duplicated
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])

# going to then create dummy encoding where each text value will have its own 
# column of 1 or 0, which is true/false
onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()

# now change and encode y column
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

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
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)

# do not need to fit_transform, already done on training set, model will need
# to accept original dataset
X_test = sc_X.transform(X_test)

# we do not need to do feature scaling to the y variable column










































