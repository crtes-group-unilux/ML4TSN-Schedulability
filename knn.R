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
  
  success <- na.omit(success)
  rownames(success) <- seq(length=nrow(success))
  return (success)
}

Factorize <- function(label){ 
  label[label == 0] <- "Feasible"
  label[label == 1] <- "Nonfeasible"
  return (label)
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
    print(paste('    FP: ', mean(falsePositives)))
    print(paste('    FN: ', mean(falseNegatives)))
    acc <- c(acc, mean(accuracies))
    FP <- c(FP, mean(falsePositives))
    FN <- c(FN, mean(falseNegatives))
  }
  return (c(acc, FP, FN))
}

allResults <- c()
print("Experiments with FIFO")
FIFO <- KNNaccuracy(labelColumn = 5) #FIFO
print("Experiments with Manual")
Manual <- KNNaccuracy(labelColumn = 6)  #Manual
print("Experiments with CP8")
CP8 <- KNNaccuracy(labelColumn = 7)  #CP8
print("Experiments with Preshaping")
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
  print(paste("acc: ", acc))
  print(paste("FP: ", FP))
  print(paste("FN: ", FN))
  #return (c(acc, FP, FN))
}

# percentage of variation should be changed manually: successVariationXpercent
# and k is the value that get the highest accuracy without variaion
print("Experiments with FIFO - variation")
KNNvariation(30,5, successVariation90percent)
print("Experiments with Manual - variation")
KNNvariation(20,6, successVariation90percent)
print("Experiments with CP8 - variation")
KNNvariation(80,7, successVariation90percent)
print("Experiments with Preshaping - variation")
KNNvariation(20,8, successVariation90percent)


##################### End of code for K-NN with variations ###############################
