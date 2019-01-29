# ML4TSN_Schedulability - Machine learning for assessing schedulability of TSN networks
# Copyright (C) 2019 University of Luxembourg
# Authors: Tieu Long Mai, Nicolas Navet
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as
# published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.


############## Supervised learning with k-NN ############################
set.seed(123)
library(ineq)
library(caret)
library(class)
library(Rmisc)


CleanData <- function(link, ifFASTanalysis){
  data <- read.csv2(link, sep='\t', header = TRUE,stringsAsFactors = TRUE)
  success <- data
  
  table(success$totalIndividualFlows)
  success <- success[success$ManualClassification < 10000,]
  # shuffle the data
  success <- success[sample(nrow(success)), ]
  # reset the row index
  rownames(success) <- seq(length=nrow(success))
  table(success$totalIndividualFlows)
  
  success$FIFO[success$FIFO > 0] <- 1
  success$RandomClassification[success$RandomClassification > 0] <- 1
  success$ManualClassification[success$ManualClassification > 0] <- 1
  success$ConcisePriorities3classes[success$ConcisePriorities3classes > 0] <- 1
  success$ConcisePriorities8classes[success$ConcisePriorities8classes > 0] <- 1
  success$solutionsOfPreshaping[success$solutionsOfPreshaping > 0] <- 2
  success$solutionsOfPreshaping[success$solutionsOfPreshaping == 0] <- 1
  success$solutionsOfPreshaping[success$solutionsOfPreshaping == 2] <- 0
  
  if (ifFASTanalysis == TRUE) {
    success$countFailsFIFOFASTanalysis[success$countFailsFIFOFASTanalysis > 0] <- 1
    success$countFailsManualClassificationFASTanalysis[success$countFailsManualClassificationFASTanalysis > 0] <- 1
    success$countFails8classesFASTanalysis[success$countFails8classesFASTanalysis > 0] <- 1
  }
  
  # Gini index
  success["GiniIndex"] <- 0
  for (i in 1:nrow(success)){
    success$GiniIndex[i] <- ineq(success[i, 197:205])
  }
  #success$GiniIndex <- ineq(success[i, 197:205])
  
  success <- na.omit(success)
  rownames(success) <- seq(length=nrow(success))
  return (success)
}

Factorize <- function(label){ 
  label[label == 0] <- "Feasible"
  label[label == 1] <- "Nonfeasible"
  #label <- as.factor(label)
  #label <- factor(label, levels = c(0, 1), labels = c("Feasible", "Nonfeasible"))
  return (label)
}

#library("gmodels")

CalculateAccuraciesKNN <- function(train, test, dataColumns, labelColumn, FASTanalysisColumn){
  
  # KNN prediction
  KforMaxAccuracy <- 0
  maxAccuracy <- 0
  upperBoundAtMaxAccuracy <- 0
  lowerBoundAtMaxAccuracy <- 0
  # false positive: non-feasible config that is wrongly predicted as feasible
  meanFalsePositivesAtMaxAccuracy <- c()
  upperBoundFalsePositivesAtMaxAccuracy <- c()
  lowerBoundFalsePositivesAtMaxAccuracy <- c()
  # false negative: feasible config that is wrongly predicted as non-feasible
  meanFalseNegativesAtMaxAccuracy <- c()
  upperBoundFalseNegativesAtMaxAccuracy <- c()
  lowerBoundFalseNegativesAtMaxAccuracy <- c()
  # testing time of K-NN
  meanTestingTimeKNNAtMaxAccuracy <- c()
  upperBoundTestingTimeKNNAtMaxAccuracy <- c()
  lowerBoundTestingTimeKNNAtMaxAccuracy <- c()
  
  sequenceOfNeighbors <- seq(10,100, by = 10)
  #sequenceOfNeighbors <- c(20)
  numberOfLoops = 100
  
  # create training and test data
  data_train = train[, c(2:5, 212)]
  data_test  = test[, c(2:5, 217)]
  
  # min max scale of training set and testing set
  minMaxScaler <- caret::preProcess(data_train, method = "range")
  data_train <- predict(minMaxScaler, data_train)
  data_test <- predict(minMaxScaler, data_test)
  
  # create labels for training and test data
  train_labels <- Factorize(train[, labelColumn])
  test_labels  <- Factorize(test[, labelColumn])
  
  for (k in sequenceOfNeighbors){ # run k-nn for some values of neighbours
    accuracies <- c()
    falsePositives <- c()
    falseNegatives <- c()
    testingTimes <- c()
    
    # KNN with 10 loops to find the max accuracy
    for(loop in 1:numberOfLoops){
      
      #start_time <- Sys.time()
      test_pred <- knn(train = data_train, test = data_test, cl = train_labels, k = k)
      #end_time <- Sys.time()
      #print(paste("time: ", (end_time - start_time)))
      
      accuracy <- sum(as.integer(test_pred == test_labels)) / length(test_labels)
      accuracies <- c(accuracies, accuracy)
      
      falsePositive <- sum(as.integer( (test_pred != test_labels) & (test_pred == "Feasible") ))
      falsePositives <- c(falsePositives, falsePositive)
      
      falseNegative <- sum(as.integer( (test_pred != test_labels) & (test_pred == "Nonfeasible") ))
      falseNegatives <- c(falseNegatives, falseNegative)
      
      testingTime <- end_time - start_time
      testingTimes <- c(testingTimes, as.double(testingTime))
    }
    
    ci <- CI(accuracies, ci=0.95)
    print(paste("Average prediction accuracy with k =", k, ":", ci[2]))
    #print("") # empty line
    if (ci[2] > maxAccuracy){
      upperBoundAtMaxAccuracy <- ci[1]
      maxAccuracy <- ci[2]
      lowerBoundAtMaxAccuracy <- ci[3]
      KforMaxAccuracy <- k
      
      ci_falsePositives <- CI(falsePositives, ci=0.95)
      upperBoundFalsePositivesAtMaxAccuracy <- ci_falsePositives[1]
      meanFalsePositivesAtMaxAccuracy <- ci_falsePositives[2]
      lowerBoundFalsePositivesAtMaxAccuracy <- ci_falsePositives[3]
      
      ci_falseNegatives <- CI(falseNegatives, ci=0.95)
      upperBoundFalseNegativesAtMaxAccuracy <- ci_falseNegatives[1]
      meanFalseNegativesAtMaxAccuracy <- ci_falseNegatives[2]
      lowerBoundFalseNegativesAtMaxAccuracy <- ci_falseNegatives[3]
      
      ci_testingTimes <- CI(testingTimes, ci=0.95)
      upperBoundTestingTimeKNNAtMaxAccuracy <- ci_testingTimes[1]
      meanTestingTimeKNNAtMaxAccuracy <- ci_testingTimes[2]
      lowerBoundTestingTimeKNNAtMaxAccuracy <- ci_testingTimes[3]
      
    }
    
    # FAST analysis 
    maxAccuracyFASTanalysis <- 0
    meanFalsePositivesFASTanalysis <- c()
    meanFalseNegativesFASTanalysis <- c()
    
    test_pred <- Factorize(test[, FASTanalysisColumn])
    
    maxAccuracyFASTanalysis <- sum(as.integer(test_pred == test_labels)) / length(test_labels)
    meanFalsePositivesFASTanalysis <- sum(as.integer( (test_pred != test_labels) & (test_pred == "Feasible") ))
    meanFalseNegativesFASTanalysis <- sum(as.integer( (test_pred != test_labels) & (test_pred == "Nonfeasible") ))
  }
  print(paste("Maximum of prediction accuracy :", maxAccuracy, " at k = ", KforMaxAccuracy ))
  return (c(c(upperBoundAtMaxAccuracy, maxAccuracy, lowerBoundAtMaxAccuracy)*100,
            c(upperBoundFalsePositivesAtMaxAccuracy, meanFalsePositivesAtMaxAccuracy, lowerBoundFalsePositivesAtMaxAccuracy),
            c(upperBoundFalseNegativesAtMaxAccuracy, meanFalseNegativesAtMaxAccuracy, lowerBoundFalseNegativesAtMaxAccuracy),
            c(upperBoundTestingTimeKNNAtMaxAccuracy, meanTestingTimeKNNAtMaxAccuracy, lowerBoundTestingTimeKNNAtMaxAccuracy),
            c(maxAccuracyFASTanalysis*100, meanFalsePositivesFASTanalysis, meanFalseNegativesFASTanalysis) ) )
}

KNNwithVariation <- function(train, test) {
  trainColumns <- c(2:5, 212)
  print("kNN for FIFO")
  KnnFIFO <- CalculateAccuraciesKNN(train, test, trainColumns, 10, FASTanalysisColumn = 214)
  print("kNN for Manual classification")
  KnnManual <- CalculateAccuraciesKNN(train, test, trainColumns, 12, FASTanalysisColumn = 215)
  print("kNN for Concise Priorities 8 classes")
  KnnCP8 <- CalculateAccuraciesKNN(train, test, trainColumns, 14, FASTanalysisColumn = 216)
  print("kNN for Preshaping")
  KnnPreshaping <- CalculateAccuraciesKNN(train, test, trainColumns, 208, FASTanalysisColumn = 216)
  return (c(KnnFIFO, KnnManual, KnnCP8, KnnPreshaping))
}

success <- CleanData(link = "TrainingDataKNN.txt", FALSE)
# remove number of flows is 10
success <- success[success$totalIndividualFlows != 10, ]
# shuffle the data - only shuffle the training data
success <- success[sample(nrow(success)), ]
# reset the row index
rownames(success) <- seq(1:nrow(success))

sequenceOfVariation <- c(0,10,20,30,40,50,60,70,80,90)
successVariation0percent <- CleanData(link = "TestingDataVariation0percent.txt", TRUE)
successVariation0percent <- successVariation0percent[successVariation0percent$totalIndividualFlows != 10, ]
successVariation10percent <- CleanData(link = "TestingDataVariation10percent.txt", TRUE)
successVariation10percent <- successVariation10percent[successVariation10percent$totalIndividualFlows != 10, ]
successVariation20percent <- CleanData(link = "TestingDataVariation20percent.txt", TRUE)
successVariation20percent <- successVariation20percent[successVariation20percent$totalIndividualFlows != 10, ]
successVariation30percent <- CleanData(link = "TestingDataVariation30percent.txt", TRUE)
successVariation30percent <- successVariation30percent[successVariation30percent$totalIndividualFlows != 10, ]
successVariation40percent <- CleanData(link = "TestingDataVariation40percent.txt", TRUE)
successVariation40percent <- successVariation40percent[successVariation40percent$totalIndividualFlows != 10, ]
successVariation50percent <- CleanData(link = "TestingDataVariation50percent.txt", TRUE)
successVariation50percent <- successVariation50percent[successVariation50percent$totalIndividualFlows != 10, ]
successVariation60percent <- CleanData(link = "TestingDataVariation60percent.txt", TRUE)
successVariation60percent <- successVariation60percent[successVariation60percent$totalIndividualFlows != 10, ]
successVariation70percent <- CleanData(link = "TestingDataVariation70percent.txt", TRUE)
successVariation70percent <- successVariation70percent[successVariation70percent$totalIndividualFlows != 10, ]
successVariation80percent <- CleanData(link = "TestingDataVariation80percent.txt", TRUE)
successVariation80percent <- successVariation80percent[successVariation80percent$totalIndividualFlows != 10, ]
successVariation90percent <- CleanData(link = "TestingDataVariation90percent.txt", TRUE)
successVariation90percent <- successVariation90percent[successVariation90percent$totalIndividualFlows != 10, ]



########################### 3D plot ######################################
library(plotly)
temp_data <- success[1:1000, ]
plot_ly(temp_data, showlegend = FALSE, showscale = FALSE, x = ~critical, y = ~audio, 
        z = ~video, color = ~factor(solutionsOfPreshaping), colors = c('green', 'red'), size = 1)
########################### End of code for 3D plot ######################



############### Find the optimal size of training set ####################
success10 <- success[1:(nrow(success)*7/8), ]
KNNwithVariation(success10, successVariation0percent)

# perform 5-folds to find how many training data we need
temp1 <- success[, c(2:5,208, 212)]
temp2 <- successVariation0percent[ , c(2:5, 208, 217)]
temp_data <- rbind(temp1, temp2)
# reset the row index
temp_data <- temp_data[sample(nrow(temp_data)), ]
rownames(temp_data) <- seq(length=nrow(temp_data))

y <- c()
for (percent in 1:10){
  temp <- temp_data[1:(nrow(temp_data)*percent/10), ]
  
  tmp = seq(1, nrow(temp))
  folds <- cut(tmp, breaks=5, labels=FALSE)
  accuracies <- c()
  for (loop in 1:10) {
    for (i in 1:5) {
      testIndexes <- which(folds==i, arr.ind=TRUE)
      # create training and test data
      data_train = temp[-testIndexes, c(1:4,6)]
      data_test  = temp[testIndexes, c(1:4,6)]
      
      # min max scale of training set and testing set
      minMaxScaler <- caret::preProcess(data_train, method = "range")
      data_train <- predict(minMaxScaler, data_train)
      data_test <- predict(minMaxScaler, data_test)
      
      # create labels for training and test data
      train_labels <- Factorize(temp[-testIndexes, 5])
      test_labels  <- Factorize(temp[testIndexes, 5])
      
      test_pred <- knn(train = data_train, test = data_test, cl = train_labels, k = 20, use.all =  TRUE)
      accuracy <- sum(as.integer(test_pred == test_labels)) / length(test_labels)
      accuracies <- c(accuracies, accuracy)
    }
  }
  print(paste("Percent: ", percent, "Average prediction accuracy k-fold:", mean(accuracies)) )
  y <- c(y, mean(accuracies))
}
x <- seq(400, 4000, by=400)
plot(x,y, type = 'b', xlab = "Size of training set.", ylab = "Accuracy of K-NN prediction with Preshaping.", 
     xaxp = c(0, 4000, 10), yaxp = c(0.9, 0.93, 6))

# results from my computer
x <- seq(400, 4000, by=400)
y <- c(89.92,	90.76,	91.82,	92.22,	92.52,	92.74,	92.53,	92.86,	93.14,	93.12)
plot(x,y, type = 'b', xlab = "Size of training set.", ylab = "Accuracy of k-NN prediction with Pre-shaping.", 
     xaxp = c(0, 4000, 10))
############## End of code for finding optimal size of training set #########################



###################### K-NN with 5-fold - No variation ######################################
temp1 <- success[, c(2:5,10,12,14,208, 212)]
temp2 <- successVariation0percent[ , c(2:5, 10,12,14,208, 217)]
temp_data <- rbind(temp1, temp2)
# reset the row index
temp_data <- temp_data[sample(nrow(temp_data)), ]
rownames(temp_data) <- seq(length=nrow(temp_data))

sequenceOfNeighbors <- seq(10,100, by=10)
temp <- temp_data
KNNaccuracy <- function (labelColumn) {
  acc <- c()
  FP <- c()
  FN <- c()
  for (k in sequenceOfNeighbors) {  
    tmp = seq(1, nrow(temp))
    folds <- cut(tmp, breaks=5, labels=FALSE)
    accuracies <- c()
    falsePositives <- c()
    falseNegatives <- c()
    for (loop in 1:10) {
      for (i in 1:5) {
        testIndexes <- which(folds==i, arr.ind=TRUE)
        # create training and test data
        data_train = temp[-testIndexes, c(1:4,9)]
        data_test  = temp[testIndexes, c(1:4,9)]
        
        # min max scale of training set and testing set
        minMaxScaler <- caret::preProcess(data_train, method = "range")
        data_train <- predict(minMaxScaler, data_train)
        data_test <- predict(minMaxScaler, data_test)
        
        # create labels for training and test data
        train_labels <- Factorize(temp[-testIndexes, labelColumn])
        test_labels  <- Factorize(temp[testIndexes, labelColumn])
        
        test_pred <- knn(train = data_train, test = data_test, cl = train_labels, k = k, use.all =  TRUE)
        accuracy <- sum(as.integer(test_pred == test_labels)) / length(test_labels)
        
        falsePositive <- sum(as.integer( (test_pred != test_labels) & (test_pred == "Feasible") ))
        falsePositives <- c(falsePositives, falsePositive)
        
        falseNegative <- sum(as.integer( (test_pred != test_labels) & (test_pred == "Nonfeasible") ))
        falseNegatives <- c(falseNegatives, falseNegative)
        
        accuracies <- c(accuracies, accuracy)
      }
    }
    print(paste('k: ', k, 'acc: ', mean(accuracies)))
    acc <- c(acc, mean(accuracies))
    FP <- c(FP, mean(falsePositives))
    FN <- c(FN, mean(falseNegatives))
  }
  return (c(acc, FP, FN))
}

allResults <- c()
FIFO <- KNNaccuracy(labelColumn = 5) #FIFO
Manual <- KNNaccuracy(labelColumn = 6)  #Manual
CP8 <- KNNaccuracy(labelColumn = 7)  #CP8
Preshaping <- KNNaccuracy(labelColumn = 8)  #Preshaping

#allResults <- rbind (FIFO, Manual, CP8, Preshaping)
#write.csv(allResults, "AllResultsKNN.txt")
###################### End of code for K-NN with 5-fold - No variation ######################################



##################### K-NN with variation ###############################
# K-NN with 5-fold
KNNvariation <- function (k, labelColumn, testing) {
  temp1 <- success[, c(2:5,10,12,14,208, 212)]
  temp2 <- testing[ , c(2:5, 10,12,14,208, 217)]
  acc <- c()
  FP <- c()
  FN <- c()
  accuracies <- c()
  falsePositives <- c()
  falseNegatives <- c()
  for (loop in 1:10) {
    # create training and test data
    data_train = temp1[ , c(1:4,9)]
    data_test  = temp2[ , c(1:4,9)]
    
    # min max scale of training set and testing set
    minMaxScaler <- caret::preProcess(data_train, method = "range")
    data_train <- predict(minMaxScaler, data_train)
    data_test <- predict(minMaxScaler, data_test)
    
    # create labels for training and test data
    train_labels <- Factorize(temp1[ , labelColumn])
    test_labels  <- Factorize(temp2[ , labelColumn])
    
    test_pred <- knn(train = data_train, test = data_test, cl = train_labels, k = k, use.all =  TRUE)
    accuracy <- sum(as.integer(test_pred == test_labels)) / length(test_labels)
    
    falsePositive <- sum(as.integer( (test_pred != test_labels) & (test_pred == "Feasible") ))
    falsePositives <- c(falsePositives, falsePositive)
    
    falseNegative <- sum(as.integer( (test_pred != test_labels) & (test_pred == "Nonfeasible") ))
    falseNegatives <- c(falseNegatives, falseNegative)
    
    accuracies <- c(accuracies, accuracy)
  }
  acc <- c(acc, mean(accuracies))
  FP <- c(FP, mean(falsePositives))
  FN <- c(FN, mean(falseNegatives))
  print(acc)
  print(FP)
  print(FN)
  #return (c(acc, FP, FN))
}

# percentage of variation should be changed manually: successVariationXpercent
KNNvariation(30,5, successVariation90percent)
KNNvariation(20,6, successVariation90percent)
KNNvariation(80,7, successVariation90percent)
KNNvariation(20,8, successVariation90percent)


##################### End of code for K-NN with variations ###############################
