# Data Preprocessing Template

# Importing the dataset
dataset = read.csv('Position_Salaries.csv')

# only keep the 2nd and 3rd columns
dataset = dataset[2:3]

# Splitting the dataset into the Training set and Test set
# install.packages('caTools')
# library(caTools)
# set.seed(123)
# split = sample.split(dataset$Salary, SplitRatio = 0.8)
# training_set = subset(dataset, split == TRUE)
# test_set = subset(dataset, split == FALSE)

# Feature Scaling
# training_set = scale(training_set)
# test_set = scale(test_set)

# fitting the linear regression to the dataset
lin_reg = lm(formula = Salary ~ ., data = dataset)

# fitting polynomial regression to the dataset
# add new independent variable column that will be the level column values to the power of 2
# the new column will allow the regression model to be built as a polynomial regression 
# model that has different levels of degrees of freedom
# you can tell if a degree of freedom level is better by running summary(poly_reg)
# which will tell you the statistics of the model, the same as before the lower the P-value
# the higher the significance level of the independent variable columns 
dataset$Level2 = dataset$Level^2
dataset$Level3 = dataset$Level^3
poly_reg = lm(formula = Salary ~ ., data = dataset)

# visualising the linear regression results
# install.packages("ggplot2")
#library(ggplot2)

# visualising the polynomial regression results










