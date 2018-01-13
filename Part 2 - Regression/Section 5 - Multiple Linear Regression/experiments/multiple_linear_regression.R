# Multiple Linear Regression

# Importing the dataset
dataset = read.csv('50_Startups.csv')

# encode categorical data for state columns
dataset$State = factor(dataset$State, 
                         levels = c('New York', 'California', 'Florida'),
                         labels = c(1,2,3))

# Splitting the dataset into the Training set and Test set
# install.packages('caTools')
library(caTools)
set.seed(123)
split = sample.split(dataset$Profit, SplitRatio = 0.8)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

# Feature Scaling
# training_set = scale(training_set)
# test_set = scale(test_set)

# fitting the multiple linear regression to the training set
# Dependent Variable ~ linear combination of Independent Variables
regressor = lm(formula = Profit ~ ., data = training_set)

# - R handles dummy variable trap so you dont need to take away a column
# - use this command in console to see statistics on regressor
# summary(regressor)

# - the lower P-value is, the more effect the independent variable has on
# a dependent variable
# - the more stars means the more significant level the variable is
# - everything above 0.05 or 0.1, it is borderlining how significant it is
# - 1 means no significant effect on the dependent variable
# - here in this data only R&D spending has a lower enough significance level
# to impact the profit dependent variable

# predicting the test set results
y_pred = predict(regressor, newdata = test_set)

# Building the Optimal model using Backward Elimination
# - removing insignificant columns/independent variables will increase the 
# value of the Adjusted R-squared value
# - Adjusted R-squared compares the explanatory power of regression models
# that contain different numbers of predictors/independent variables
# - Adjusted R-squared value increase only if the new term/predictor improves
# the model more than would be expected by chance
# - Adjusted R-squared value decrease only if the predictor improves the 
# model by less than expected by chance
# - it is always lower than the R-squared
# - R-Squared: http://blog.minitab.com/blog/adventures-in-statistics-2/multiple-regession-analysis-use-adjusted-r-squared-and-predicted-r-squared-to-include-the-correct-number-of-variables
# Step 1 - Select a significance level to stay in the model - we will use 0.05 of SL
# Step 2 - Come up with initial model with all independent variables and dataset
regressor = lm(formula = Profit ~ R.D.Spend + Administration + Marketing.Spend + State, 
               data = dataset)
summary(regressor)

# Step 3 - Remove the predictor with the highest P-value if it is > SL, 
# this is Step 4 (the removal portion)
# Step 5 - Fit model without this variable by regenerating model 
regressor = lm(formula = Profit ~ R.D.Spend + Administration + Marketing.Spend, 
               data = dataset)
summary(regressor)

# remove administration info since its P-value is higher than SL of 0.05
# then rebuild regressor without it
regressor = lm(formula = Profit ~ R.D.Spend + Marketing.Spend, 
               data = dataset)
summary(regressor)

# remove marketing spend info since its P-value is higher than SL of 0.05
# then rebuild regressor without it
# however, it is a SL of 0.1, so its borderline whether we need to, but 
# because we chose 0.05 we decide to remove it for now
regressor = lm(formula = Profit ~ R.D.Spend, data = dataset)
summary(regressor)

# this is the final model which determines that only R.S.Spend is a significant
# independent variable towards the dependent variable of Profit