# categorical data

dataset = read.csv("Data.csv")

# take care of missing data
# replaces missing values with the mean of the column if there are values in the column
# combines with the values that are there, does not replace those
dataset$Age = ifelse(is.na(dataset$Age), 
                     ave(dataset$Age, FUN=function(x) mean(x, na.rm = TRUE)), dataset$Age
)

dataset$Salary = ifelse(is.na(dataset$Salary), 
                        ave(dataset$Salary, FUN=function(x) mean(x, na.rm = TRUE)), dataset$Salary
)

# encode categorical data
dataset$Country = factor(dataset$Country, 
                         levels = c('France', 'Spain', 'Germany'),
                         labels = c(1,2,3))

dataset$Purchased = factor(dataset$Purchased, 
                         levels = c('No', 'Yes'),
                         labels = c(0, 1))

# splitting the dataset into the training set and test set
# install library, comment out second time only need to run line once
# install.packages("caTools")

# import library
library(caTools)
set.seed(123)

# split dataset and split amount for training set
split = sample.split(dataset$Purchased, SplitRatio = 0.8)

# condition acts as a check to see whether we need to split to a particular set
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

# feature scaling - do not scale columns with factors as they are not numeric
training_set[, 2:3] = scale(training_set[, 2:3])
test_set[, 2:3] = scale(test_set[, 2:3])






