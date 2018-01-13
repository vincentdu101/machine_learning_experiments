# missing data

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