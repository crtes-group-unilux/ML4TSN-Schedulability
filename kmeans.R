
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

  
############## Unsupervised learning with K-means ############################
set.seed(123)

library(ineq)
library("caret")
library("class")
library(Rmisc)

lastColumn <- length(names(success))
# get the training data from kNN 
dataVoting <- success[1:500 , c(2:5, 10, 12, 14, 208, lastColumn)]
rownames(dataVoting) <- seq(length=nrow(dataVoting))

# create normalization function
CleanData <- function(link){
  data <- read.csv2(link, sep='\t', header = TRUE,stringsAsFactors = TRUE)
  success <- data
  
  success <- success[success$totalIndividualFlows != 10, ]
  
  #table(success$totalIndividualFlows)
  success <- success[success$maxLoad < 100, ]
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
  
  # Gini index
  success["GiniIndex"] <- 0
  for (i in 1:nrow(success)){
    success$GiniIndex[i] <- ineq(success[i, 197:205])
  }
  
  success <- na.omit(success)
  rownames(success) <- seq(length=nrow(success))
  
  lastColumn <- length(names(success))
  
  return (success[ , c(2:5, 10, 12, 14, 208, lastColumn)])
}

sequenceOfNeighbors <- c(1,10,20,30,40,50,60,70,80,90,100)
numberOfLoops = 10
numberOfKmeansClusters = 2

Factorize <- function(label){ 
  label[label == 0] <- "Feasible"
  label[label == 1] <- "Nonfeasible"
  #label <- as.factor(label)
  #label <- factor(label, levels = c(0, 1), labels = c("Feasible", "Nonfeasible"))
  return (label)
}

PredictionwithVariation <- function(train, test) {
  #trainColumns <- c(1:4,10)
  trainColumns <- c(1:4,9)
  print("kmeans for FIFO")
  KmeansFIFO <- CalculateAccuraciesKmeans(train, test, trainColumns, 5)
  print("Kmeans for Manual classification")
  KmeansManual <- CalculateAccuraciesKmeans(train, test, trainColumns, 6)
  print("kmeans for Concise Priorities 8 classes")
  KmeansCP8 <- CalculateAccuraciesKmeans(train, test, trainColumns, 7)
  print("kmeans for Preshaping")
  KmeansPreshaping <- CalculateAccuraciesKmeans(train, test, trainColumns, 9)
  return (c(KmeansFIFO, KmeansManual, KmeansCP8, KmeansPreshaping))
}

training_set1 <- CleanData(link = "TrainingDataKmeans1.txt")
training_set2 <- CleanData(link = "TrainingDataKmeans2.txt")

training_set <- rbind(training_set1, training_set2)

# shuffle the data - only shuffle the training data
training_set <- training_set[sample(nrow(training_set)), ]
# reset the row index
rownames(training_set) <- seq(1:nrow(training_set))



# some columns only have value 1, we fix that by adding some values of 1 and 0
if (TRUE) {
  training_set$FIFO[1:10] <- 0
  training_set$FIFO[11:20] <- 1
  training_set$ManualClassification[1:10] <- 0
  training_set$ManualClassification[11:20] <- 1
  training_set$ConcisePriorities8classes[1:10] <- 0
  training_set$ConcisePriorities8classes[11:20] <- 1
  training_set$solutionsOfPreshaping[1:10] <- 0
  training_set$solutionsOfPreshaping[11:20] <- 1
}

minMaxScaler <- caret::preProcess(training_set, method = "range")

training_set <- predict(minMaxScaler, training_set)

sequenceOfVariation <- c(0,10,20,30,40,50,60,70,80,90)
successVariation0percent <- CleanData(link = "TestingDataVariation0percent.txt")
successVariation10percent <- CleanData(link = "TestingDataVariation10percent.txt")
successVariation20percent <- CleanData(link = "TestingDataVariation20percent.txt")
successVariation30percent <- CleanData(link = "TestingDataVariation30percent.txt")
successVariation40percent <- CleanData(link = "TestingDataVariation40percent.txt")
successVariation50percent <- CleanData(link = "TestingDataVariation50percent.txt")
successVariation60percent <- CleanData(link = "TestingDataVariation60percent.txt")
successVariation70percent <- CleanData(link = "TestingDataVariation70percent.txt")
successVariation80percent <- CleanData(link = "TestingDataVariation80percent.txt")
successVariation90percent <- CleanData(link = "TestingDataVariation90percent.txt")

# do min max scale based one the scale of the training set
successVariation0percent <- predict(minMaxScaler, successVariation0percent)
successVariation10percent <- predict(minMaxScaler, successVariation10percent)
successVariation20percent <- predict(minMaxScaler, successVariation20percent)
successVariation30percent <- predict(minMaxScaler, successVariation30percent)
successVariation40percent <- predict(minMaxScaler, successVariation40percent)
successVariation50percent <- predict(minMaxScaler, successVariation50percent)
successVariation60percent <- predict(minMaxScaler, successVariation60percent)
successVariation70percent <- predict(minMaxScaler, successVariation70percent)
successVariation80percent <- predict(minMaxScaler, successVariation80percent)
successVariation90percent <- predict(minMaxScaler, successVariation90percent)


######### Voting for K-means ##############################
# New K-means accuracy function here
CalculateAccuraciesKmeans <- function(train, test, dataVoting, dataColumns, labelColumn){
  
  # create training and test data
  data_train = train[, dataColumns]
  data_test  = test[, dataColumns]
  dataVoting_data <- dataVoting[, dataColumns]
  
  # create labels for test data
  test_labels  <- Factorize(test[, labelColumn])
  datavoting_labels <- Factorize(dataVoting[, labelColumn])
  
  acc <- c()
  FP <- c()
  FN <- c()
  TP <- c()
  TN <- c()
  
  numberOfClusters <- seq(2,20,by=2)
  
  for(k in numberOfClusters){
    
    accuracies <- c()
    falsePositives <- c()
    falseNegatives <- c()
    
    truePositives <- c()
    trueNegatives <- c()
    
    for (loop in 1:10) {
      
      cl <- kmeans(data_train, k, nstart = 1, iter.max = 50)
      cl_test <- cl_predict(cl, data_test)
      
      cl_voting <- cl_predict(cl, dataVoting_data)
      # add label for each cluster
      cluster_label <- c()
      for (i in 1:k) {
        cluster_indexes <- which(cl_voting == i)
        voting_labels  <- datavoting_labels[cluster_indexes]
        
        x <- table(voting_labels)
        
        if(is.na(x["Feasible"]) == TRUE) { x["Feasible"] <- 0}
        if(is.na(x["Nonfeasible"]) == TRUE) { x["Nonfeasible"] <- 0}
        
        if (x["Nonfeasible"] > x["Feasible"]) {
          # this cluster has less critical flows than average
          cluster_label <- c(cluster_label, "Nonfeasible")
        } else {
          cluster_label <- c(cluster_label, "Feasible")
        }
      }
      
      # compare cluster label with test set
      accuracy <- 0
      falsePositive <- 0
      falseNegative <- 0
      truePositive <- 0
      trueNegative <- 0
      for (i in 1:k) {
        test_cluster_indexes <- which(cl_test == i)
        test_cluster_label <- test_labels[test_cluster_indexes]
        
        test_pred <- 1:length(test_cluster_label)
        test_pred[] <- cluster_label[i]
        
        # R automatically convert integer to String, hence we can make comparison
        accuracy <- accuracy + sum(as.integer(test_pred == test_cluster_label))
        #print(paste(sum(as.integer(test_pred == test_cluster_label)), length(test_cluster_indexes)))
        
        falsePositive <- falsePositive + sum(as.integer((test_pred != test_cluster_label) & (test_pred == "Feasible") ))
        
        falseNegative <- falseNegative + sum(as.integer( (test_pred != test_cluster_label) & (test_pred == "Nonfeasible") ))
        
        truePositive <- falsePositive + sum(as.integer((test_pred == test_cluster_label) & (test_pred == "Feasible") ))
        
        trueNegative <- falseNegative + sum(as.integer( (test_pred == test_cluster_label) & (test_pred == "Nonfeasible") ))
      }
      
      accuracy <- accuracy / length(test_labels)
      
      accuracies <- c(accuracies, mean(accuracy))
      falsePositives <- c(falsePositives, mean(falsePositive))
      falseNegatives <- c(falseNegatives, mean(falseNegative))
      
      truePositives <- c(truePositives, mean(truePositive))
      trueNegatives <- c(trueNegatives, mean(trueNegative))
    }
    acc <- c(acc, mean(accuracies))
    FP <- c(FP, mean(falsePositives))
    FN <- c(FN, mean(falseNegatives))
    TP <- c(TP, mean(truePositives))
    TN <- c(TN, mean(trueNegatives))
  }
  return (c(acc, FP, FN, TP, TN))
}

dataVoting <- predict(minMaxScaler, dataVoting)

dataColumns <- c(1:4,9)
labelColumn <- 7
results <- CalculateAccuraciesKmeans(training_set, successVariation0percent, dataVoting, dataColumns, labelColumn)

results[1:10]*100
results[11:20]/10
results[21:30]/10
# calculate Kappa
a <- results[31:40] # TP
b <- results[11:20] # FP
c <- results[21:30] # FN
d <- results[41:50] # TN
acc <- results[1:10]

pE <- ((a+b)*(a+c) + (c+d)*(b+d))/(a+b+c+d) / 1000
kappa <- (acc - pE) / (1 - pE)
kappa


################## Accuracy versus size of voting set ##############################
x <- seq(100, 1000, by=100)
y <- c(85.46,	86.37,	88.55,	89.35,	89.35,	89.31,	89.53,	89.71,	89.67,	89.49)
plot(x,y, type = 'b', xlab = "Size of voting set", ylab = "Accuracy of K-means (K = 20) with Preshaping (%)", 
     xaxp = c(100, 1000, 9))
################## End of code for Accuracy versus size of voting set ##############################



####################### Elbow Method ##############################
k.max <- 100
dataColumns <- c(1:4,9)
data_train <- training_set[, dataColumns]

wss <- sapply(1:k.max, function(k){kmeans(data_train, k, nstart=1,iter.max = 10 )$tot.withinss})

plot(1:k.max, wss,
     type="b", pch = 19, frame = FALSE, 
     xlab="Number of clusters K",
     ylab="Total within-clusters sum of squared distances")
####################### End of code for Elbow Method ##############################



############ variation K-means #########################
lastColumn <- length(names(success))
dataVoting <- success[1:500 , c(2:5, 10, 12, 14, 208, lastColumn)]
rownames(dataVoting) <- seq(length=nrow(dataVoting))
dataVoting <- predict(minMaxScaler, dataVoting)

dataColumns <- c(1:4,9)
results <- CalculateAccuraciesKmeans(training_set, successVariation90percent, dataVoting, dataColumns, 5, 20)
results
results <- CalculateAccuraciesKmeans(training_set, successVariation90percent, dataVoting, dataColumns, 6, 12)
results
results <- CalculateAccuraciesKmeans(training_set, successVariation90percent, dataVoting, dataColumns, 7, 10)
results
results <- CalculateAccuraciesKmeans(training_set, successVariation90percent, dataVoting, dataColumns, 8, 12)
results


