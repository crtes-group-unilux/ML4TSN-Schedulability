set.seed(123)
library(ineq)
library(caret)
library(class)
library(Rmisc)

# create normalization function
normalize <- function(x) {
  return ((x - min(x)) / (max(x) - min(x)))
}

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
    success$GiniIndex[i] <- ineq(success[i, 217:252])
  }
  
  success <- na.omit(success)
  rownames(success) <- seq(length=nrow(success))
  # success <- lapply(success, normalize)
  # success <- as.data.frame(success)
  
  return (success)
}

Factorize <- function(label){ 
  label[label == 0] <- "Feasible"
  label[label == 1] <- "Nonfeasible"
  #label <- as.factor(label)
  #label <- factor(label, levels = c(0, 1), labels = c("Feasible", "Nonfeasible"))
  return (label)
}



success <- CleanData(link = "TrainingDataKNN_newTopo.txt", TRUE)
# shuffle the data - only shuffle the training data
success <- success[sample(nrow(success)), ]
# reset the row index
rownames(success) <- seq(1:nrow(success))

sequenceOfVariation <- c(0,10,20,30,40,50,60,70,80,90)
successVariation0percent <- CleanData(link = "TestingDataVariation0percent_newTopo.txt", TRUE)

successVariation10percent <- CleanData(link = "TestingDataVariation10percent_newTopo.txt", TRUE)

successVariation20percent <- CleanData(link = "TestingDataVariation20percent_newTopo.txt", TRUE)

successVariation30percent <- CleanData(link = "TestingDataVariation30percent_newTopo.txt", TRUE)

successVariation40percent <- CleanData(link = "TestingDataVariation40percent_newTopo.txt", TRUE)

successVariation50percent <- CleanData(link = "TestingDataVariation50percent_newTopo.txt", TRUE)

successVariation60percent <- CleanData(link = "TestingDataVariation60percent_newTopo.txt", TRUE)

successVariation70percent <- CleanData(link = "TestingDataVariation70percent_newTopo.txt", TRUE)

successVariation80percent <- CleanData(link = "TestingDataVariation80percent_newTopo.txt", TRUE)

successVariation90percent <- CleanData(link = "TestingDataVariation90percent_newTopo.txt", TRUE)



###################### K-NN with 5-fold - No variation ######################################
temp1 <- success[, c(2:5,10,12,14,255, 264)]
temp2 <- successVariation0percent[ , c(2:5, 10,12,14,255, 264)]
temp_data <- rbind(temp1, temp2)
# reset the row index
temp_data <- temp_data[sample(nrow(temp_data)), ]
rownames(temp_data) <- seq(length=nrow(temp_data))

sequenceOfNeighbors <- seq(10,100, by=10)
temp <- temp_data

KNNaccuracy <- function (labelColumn) {
  acc <- c()
  TPR_all <- c()
  TNR_all <- c()
  for (k in sequenceOfNeighbors) {  
    tmp = seq(1, nrow(temp))
    folds <- cut(tmp, breaks=5, labels=FALSE)
    accuracies <- c()
    falsePositives <- c()
    falseNegatives <- c()
    TPRs <- c()
    TNRs <- c()
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
        
        test_pred <- class::knn(train = data_train, test = data_test, cl = train_labels, k = k, use.all =  TRUE)
        accuracy <- sum(as.integer(test_pred == test_labels)) / length(test_labels)
        
        FP <- sum(as.integer( (test_pred != test_labels) & (test_pred == "Feasible") ))
        FN <- sum(as.integer( (test_pred != test_labels) & (test_pred == "Nonfeasible") ))
        TP <- sum(as.integer( (test_pred == test_labels) & (test_pred == "Feasible") ))
        TN <- sum(as.integer( (test_pred == test_labels) & (test_pred == "Nonfeasible") ))
        
        TPR <- TP / (TP + FN)
        TPRs <- c(TPRs, TPR)
        
        TNR <- TN / (TN + FP)
        TNRs <- c(TNRs, TNR)
        
        accuracies <- c(accuracies, accuracy)
      }
    }
    print(paste('k: ', k, 'acc: ', mean(accuracies)))
    acc <- c(acc, mean(accuracies))
    TPR_all <- c(TPR_all, mean(TPRs))
    TNR_all <- c(TNR_all, mean(TNRs))
  }
  return (c(acc, TPR_all, TNR_all))
}

allResults <- c()
FIFO <- KNNaccuracy(labelColumn = 5) #FIFO
Manual <- KNNaccuracy(labelColumn = 6)  #Manual
CP8 <- KNNaccuracy(labelColumn = 7)  #CP8
Preshaping <- KNNaccuracy(labelColumn = 8)  #Preshaping
###################### End of code for K-NN with 5-fold - No variation ######################################



############# Kappa test ##################################
library(caret)
KNNvariation <- function (k, labelColumn, testing) {
  temp1 <- success[, c(2:5,10,12,14,255, 264)]
  temp2 <- testing[ , c(2:5, 10,12,14,255, 264)]
  acc <- c()
  FP <- c()
  FN <- c()
  accuracies <- c()
  falsePositives <- c()
  falseNegatives <- c()
  for (loop in 1:1) {
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
    
    test_pred <- class::knn(train = data_train, test = data_test, cl = train_labels, k = k, use.all =  TRUE)
    accuracy <- sum(as.integer(test_pred == test_labels)) / length(test_labels)
    
    falsePositive <- sum(as.integer( (test_pred != test_labels) & (test_pred == "Feasible") ))
    falsePositives <- c(falsePositives, falsePositive)
    
    falseNegative <- sum(as.integer( (test_pred != test_labels) & (test_pred == "Nonfeasible") ))
    falseNegatives <- c(falseNegatives, falseNegative)
    
    accuracies <- c(accuracies, accuracy)
    
    x <- caret::confusionMatrix(table(test_pred, test_labels))
    print(x)
    
  }
  acc <- c(acc, mean(accuracies))
  FP <- c(FP, mean(falsePositives))
  FN <- c(FN, mean(falseNegatives))
}

# number of clusters K is found by the previous experiments
KNNvariation(40,5, successVariation0percent)
KNNvariation(30,6, successVariation0percent)
KNNvariation(30,7, successVariation0percent)
KNNvariation(60,8, successVariation0percent)
############# End of Kappa test ##################################



##################### K-NN with variation ###############################
KNNvariation <- function (k, labelColumn, testing) {
  temp1 <- success[, c(2:5,10,12,14,255, 264)]
  temp2 <- testing[ , c(2:5, 10,12,14,255, 264)]
  accuracies <- c()
  TPRs <- c()
  TNRs <- c()
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
    
    test_pred <- class::knn(train = data_train, test = data_test, cl = train_labels, k = k, use.all =  TRUE)
    accuracy <- sum(as.integer(test_pred == test_labels)) / length(test_labels)
    
    FP <- sum(as.integer( (test_pred != test_labels) & (test_pred == "Feasible") ))
    FN <- sum(as.integer( (test_pred != test_labels) & (test_pred == "Nonfeasible") ))
    TP <- sum(as.integer( (test_pred == test_labels) & (test_pred == "Feasible") ))
    TN <- sum(as.integer( (test_pred == test_labels) & (test_pred == "Nonfeasible") ))
    
    TPR <- TP / (TP + FN)
    TPRs <- c(TPRs, TPR)
    
    TNR <- TN / (TN + FP)
    TNRs <- c(TNRs, TNR)
    
    accuracies <- c(accuracies, accuracy)
  }
  print(paste(round(mean(accuracies)*100,2), round(mean(TPRs)*100,2), round(mean(TNRs)* 100,2)))
}


KNNvariation(40,5, successVariation90percent) # FIFO
KNNvariation(30,6, successVariation90percent) # Manual
KNNvariation(30,7, successVariation90percent) # CP8
KNNvariation(60,8, successVariation90percent) # Preshaping
##################### End of code for K-NN with variation ###############################
