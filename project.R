#Load raw data
training_raw_data = read.csv("~/Coursera/Data Science/Practical Machine Learning/Project/pml-training.csv")
testing_raw_data = read.csv("~/Coursera/Data Science/Practical Machine Learning/Project/pml-testing.csv")

summary(training_data)

#Clean up data

#Remove new_window data
training_data1 = subset(training_raw_data, new_window = "yes")

#Remove sinle value, NA and other non-measurement columns
data_var <- lapply(training_data1, var, na.rm = TRUE)
dv <- (data_var != 0) & (data_var != 'NA') &
    !(names(data_var) %in% c('X', 'user_name', 'raw_timestamp_part_1', 'raw_timestamp_part_2', 'cvtd_timestamp', 'num_window'))
dv1 <- dv
names(dv1)[length(dv1)] <- "classe"
training_data2 <- training_data1[dv1]



inTrain <- createDataPartition(y=training_data2$classe,
                               p=0.7, list=FALSE)
training <- training_data2[inTrain,]
testing <- training_data2[-inTrain,]


install.packages('doSNOW')
library(doSNOW)
getDoParWorkers()
getDoParName()
registerDoSNOW(makeCluster(12, type = "SOCK")) #I'm using 7 of 8 cores available. change as needed
getDoParWorkers()
getDoParName()
library(foreach)



fit <- train(classe ~.,method="gbm",data=training)

res <- predict(fit, training)
confusionMatrix(training$classe ,res)


dv[length(dv)] <- FALSE
res1 <- predict(fit, testing_data[dv])
confusionMatrix(training$classe ,res1)
