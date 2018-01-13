# categorical data

dataset = read.csv("Data.csv")
# dataset = dataset[, 2:3]

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
# training_set[, 2:3] = scale(training_set[, 2:3])
# test_set[, 2:3] = scale(test_set[, 2:3])






