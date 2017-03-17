

# rank


dir_name <- "Data/New_total_Q12005.csv"

loan_data <- read.csv(dir_name, row.names=NULL)
# ensure results are repeatable
set.seed(7)
# load the library
library(mlbench)
library(caret)
# load the dataset
data(PimaIndiansDiabetes)
# prepare training scheme
# loan_data <- na.omit(loan_data)

lapply(myData[-1], var, na.rm = TRUE) != 0

# filter out the columns that do not have unique values > 1
myData <- Filter(function(x)(length(unique(x))>1), myData)

control <- trainControl(method="repeatedcv", number=10, repeats=3)
# train the model
model <- train(IS_DELINQUENT~., data=loan_data, method="lvq", preProcess="scale", trControl=control)
# estimate variable importance
importance <- varImp(model, scale=FALSE)
# summarize importance
print(importance)
# plot importance
plot(importance)
# select k best python scikitlearn