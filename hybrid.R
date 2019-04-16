set.seed(123)
library(ineq)
library(caret)
library(class)
library(Rmisc)
library(FNN)
library(plotly)
library(ggplot2)

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
    success$GiniIndex[i] <- ineq(success[i, 197:205])
  }
  #success$GiniIndex <- ineq(success[i, 197:205])
  
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

Reverse_Factorize <- function(label) {
  label[label == "Feasible"] <- 0
  label[label == "Nonfeasible"] <- 1
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
rownames(successVariation0percent) <- seq(1:nrow(successVariation0percent))


##################### k-NN + precise scheduling analysis ########################
train <- success[, c(2:5,10,12,14,208, 212)]
test <- successVariation0percent[ , c(2:5, 10,12,14,208, 217)]

# min max scale of training set and testing set
minMaxScaler <- caret::preProcess(train, method = "range")
train <- predict(minMaxScaler, train)
test <- predict(minMaxScaler, test)

KNNaccuracy <- function (labelColumn, k, train, test, adjust_threshold) {
  
  train_columns <- c(1:4,9)   # c(1:4,9)
  # create training and test data
  data_train = train[ , train_columns]
  data_test = test[ ,train_columns]
  
  # min max scale of training set and testing set
  #minMaxScaler <- caret::preProcess(data_train, method = "range")
  #data_train <- predict(minMaxScaler, data_train)
  #data_test <- predict(minMaxScaler, data_test)
  
  # create labels for training and test data
  train_labels <- Factorize(train[ , labelColumn])
  test_labels  <- Factorize(test[ , labelColumn])
  
  # test
  test_pred <- FNN::knn(train = data_train, test = data_test, cl = train_labels, k = k)
  accuracy <- sum(as.integer(test_pred == test_labels)) / length(test_labels)
  print(paste("accuracy of test set: ", accuracy))
  
  TP <- sum(as.integer( (test_pred == test_labels) & (test_pred == "Feasible") ))
  FP <- sum(as.integer( (test_pred != test_labels) & (test_pred == "Feasible") ))
  TN <- sum(as.integer( (test_pred == test_labels) & (test_pred == "Nonfeasible") ))
  FN <- sum(as.integer( (test_pred != test_labels) & (test_pred == "Nonfeasible") ))
  print(paste("TPR: ", (TP*100)/(TP+FN), "TNR: ", (TN*100)/(TN+FP)))
  
  test["knn_predict"] <- test_pred
  pred <- test_pred
  test_pred <- as.numeric(test_pred)
  test_pred[test_pred == 2] <- "Nonfeasible"
  test_pred[test_pred == 1] <- "Feasible"
  test["predict"] <- test_pred
  for (i in 1:length(test_labels)) {
    if(pred[i] != test_labels[i])
      test$predict[i] <- "wrong"
  }
  # find the index of nearest neighbors
  indices <- attr(pred, "nn.index")
  distances <- attr(pred, "nn.dist")
  
  # ratio of feasible / non-feasbile of nearest neighbors
  test["true_labels"] <- test_labels
  test["Feasible_neighbors"] <- 0
  test["Nonfeasible_neighbors"] <- 0
  test["Mean_distance"] <- 0
  test["Min_distance"] <- 0
  test["Max_distance"] <- 0
  for (i in 1:length(test_labels)) {
    labelsOfNeighbors <- train_labels[indices[i, ]]
    test$Feasible_neighbors[i] <- length(labelsOfNeighbors[labelsOfNeighbors == "Feasible"])
    test$Nonfeasible_neighbors[i] <- length(labelsOfNeighbors[labelsOfNeighbors == "Nonfeasible"])
    test$Mean_distance[i] <- mean(distances[i, ])
    test$Min_distance[i] <- min(distances[i, ])
    test$Max_distance[i] <- max(distances[i, ])
  }
  
  # use scheduling with some cases
  adjust_threshold <- as.integer((k*adjust_threshold)/100)
  temp <- test[(test$Feasible_neighbors >= adjust_threshold) & 
                 (test$Nonfeasible_neighbors >= adjust_threshold),]
  adjust_index <- as.numeric(rownames(temp))
  print(paste("number of scheduling cases: ", length(adjust_index)))
  test_pred[adjust_index] <- test_labels[adjust_index]
  test["new_predict"] <- test_pred
  accuracy <- sum(as.integer(test_pred == test_labels)) / length(test_labels)
  print(paste("accuracy of test set after using scheduling: ", accuracy))
  
  TP <- sum(as.integer( (test_pred == test_labels) & (test_pred == "Feasible") ))
  FP <- sum(as.integer( (test_pred != test_labels) & (test_pred == "Feasible") ))
  TN <- sum(as.integer( (test_pred == test_labels) & (test_pred == "Nonfeasible") ))
  FN <- sum(as.integer( (test_pred != test_labels) & (test_pred == "Nonfeasible") ))
  print(paste("TPR: ", (TP*100)/(TP+FN), "TNR: ", (TN*100)/(TN+FP)))
  
  return (test)
}

# labelColumns: 5(FIFO), 6(Manual), 7(CP8), 8(Preshaping)
test <- KNNaccuracy(labelColumn = 8, k = 20, train, test, adjust_threshold=5)

wrong <- test[test$predict == 'wrong',]
Feasible <- test[test$predict == 'Feasible',]
Nonfeasibe<- test[test$predict == 'Nonfeasible',]

temp <- test[(test$Feasible_neighbors >= 30) & (test$Nonfeasible_neighbors >= 30),]
temp <- temp[temp$test == 'wrong', ]


# 3D plot of prediction. Black is wrong prediction
test$critical <- successVariation0percent$critical
test$audio <- successVariation0percent$audio
test$video <- successVariation0percent$video
library(plotly)
plot_ly(test, showlegend = FALSE, showscale = FALSE, x = ~critical, y = ~audio, 
        z = ~video, color = ~factor(predict), colors = c('green', 'red','black'), size = 1)
############## End of k-NN + precise scheduling analysis ####################




##################### k-NN + approximate + precise scheduling analysis ########################
train <- success[, c(2:5,10,12,14,208, 212)]
test <- successVariation0percent[ , c(2:5, 10,12,14,208, 217)]
approximate_FIFO <- Factorize(successVariation0percent$countFailsFIFOFASTanalysis)
approximate_Manual <- Factorize(successVariation0percent$countFailsManualClassificationFASTanalysis)
approximate_CP8 <- Factorize(successVariation0percent$countFails8classesFASTanalysis)

# min max scale of training set and testing set
minMaxScaler <- caret::preProcess(train, method = "range")
train <- predict(minMaxScaler, train)
test <- predict(minMaxScaler, test)

KNNaccuracy <- function (labelColumn, k, train, test, approximate, adjust_threshold) {
  
  train_columns <- c(1:4,9)   # c(1:4,9)
  # create training and test data
  data_train = train[ , train_columns]
  data_test = test[ ,train_columns]
  
  # min max scale of training set and testing set
  #minMaxScaler <- caret::preProcess(data_train, method = "range")
  #data_train <- predict(minMaxScaler, data_train)
  #data_test <- predict(minMaxScaler, data_test)
  
  # create labels for training and test data
  train_labels <- Factorize(train[ , labelColumn])
  test_labels  <- Factorize(test[ , labelColumn])
  
  # test
  test_pred <- FNN::knn(train = data_train, test = data_test, cl = train_labels, k = k)
  accuracy <- sum(as.integer(test_pred == test_labels)) / length(test_labels)
  print(paste("accuracy of test set: ", accuracy))
  
  TP <- sum(as.integer( (test_pred == test_labels) & (test_pred == "Feasible") ))
  FP <- sum(as.integer( (test_pred != test_labels) & (test_pred == "Feasible") ))
  TN <- sum(as.integer( (test_pred == test_labels) & (test_pred == "Nonfeasible") ))
  FN <- sum(as.integer( (test_pred != test_labels) & (test_pred == "Nonfeasible") ))
  print(paste("TPR: ", (TP*100)/(TP+FN), "TNR: ", (TN*100)/(TN+FP)))
  
  test["knn_predict"] <- test_pred
  pred <- test_pred
  test_pred <- as.numeric(test_pred)
  test_pred[test_pred == 2] <- "Nonfeasible"
  test_pred[test_pred == 1] <- "Feasible"
  test["predict"] <- test_pred
  for (i in 1:length(test_labels)) {
    if(pred[i] != test_labels[i])
      test$predict[i] <- "wrong"
  }
  # find the index of nearest neighbors
  indices <- attr(pred, "nn.index")
  distances <- attr(pred, "nn.dist")
  
  # ratio of feasible / non-feasbile of nearest neighbors
  test["true_labels"] <- test_labels
  test["Feasible_neighbors"] <- 0
  test["Nonfeasible_neighbors"] <- 0
  test["Mean_distance"] <- 0
  test["Min_distance"] <- 0
  test["Max_distance"] <- 0
  for (i in 1:length(test_labels)) {
    labelsOfNeighbors <- train_labels[indices[i, ]]
    test$Feasible_neighbors[i] <- length(labelsOfNeighbors[labelsOfNeighbors == "Feasible"])
    test$Nonfeasible_neighbors[i] <- length(labelsOfNeighbors[labelsOfNeighbors == "Nonfeasible"])
    test$Mean_distance[i] <- mean(distances[i, ])
    test$Min_distance[i] <- min(distances[i, ])
    test$Max_distance[i] <- max(distances[i, ])
  }
  
  # combine with approximate and precise analysis
  test["approximate"] <- approximate
  
  adjust_threshold <- as.integer((k*adjust_threshold)/100)
  temp <- test[(test$Feasible_neighbors >= adjust_threshold) & 
                 (test$Nonfeasible_neighbors >= adjust_threshold),]
  adjust_index <- as.numeric(rownames(temp))
  print(paste("number of approximate analysis: ", length(adjust_index)))
  for (i in adjust_index) {
    if(approximate[i] == "Feasible") {
      test_pred[i] = approximate[i]
    }
  }
  temp <- temp[temp$approximate != "Feasible", ]
  adjust_index <- as.numeric(rownames(temp))
  print(paste("number of precise analysis: ", length(adjust_index)))
  test_pred[adjust_index] <- test_labels[adjust_index]
  test["new_predict"] <- test_pred
  accuracy <- sum(as.integer(test_pred == test_labels)) / length(test_labels)
  print(paste("accuracy of test set after using scheduling: ", accuracy))
  
  TP <- sum(as.integer( (test_pred == test_labels) & (test_pred == "Feasible") ))
  FP <- sum(as.integer( (test_pred != test_labels) & (test_pred == "Feasible") ))
  TN <- sum(as.integer( (test_pred == test_labels) & (test_pred == "Nonfeasible") ))
  FN <- sum(as.integer( (test_pred != test_labels) & (test_pred == "Nonfeasible") ))
  print(paste("TPR: ", (TP*100)/(TP+FN), "TNR: ", (TN*100)/(TN+FP)))
  
  return (test)
}

# labelColumns: 5(FIFO), 6(Manual), 7(CP8), 8(Preshaping)
test <- KNNaccuracy(labelColumn = 7, k = 100, 
                    train, test, approximate = approximate_CP8, adjust_threshold = 15)

wrong <- test[test$predict == 'wrong',]
Feasible <- test[test$predict == 'Feasible',]
Nonfeasibe<- test[test$predict == 'Nonfeasible',]
############## End of k-NN + approximate + precise scheduling analysis ####################



################ Plot histogram of each feature ##########################################
train <- success[, c(2:5,10,12,14,208, 212)]
test <- successVariation0percent[ , c(2:5, 10,12,14,208, 217)]

# min max scale of training set and testing set
#minMaxScaler <- caret::preProcess(train, method = "range")
#train <- predict(minMaxScaler, train)
#test <- predict(minMaxScaler, test)

color_red <- adjustcolor('red', alpha.f = 0.5) 
color_green <- adjustcolor('blue', alpha.f = 0.5)

# FIFO
Feasible <- train[train$FIFO == 0, ]
Nonfeasible <- train[train$FIFO == 1, ]

# Manual
Feasible <- train[train$ManualClassification == 0, ]
Nonfeasible <- train[train$ManualClassification == 1, ]

# CP8
Feasible <- train[train$ConcisePriorities8classes == 0, ]
Nonfeasible <- train[train$ConcisePriorities8classes == 1, ]

# Preshaping
Feasible <- train[train$solutionsOfPreshaping == 0, ]
Nonfeasible <- train[train$solutionsOfPreshaping == 1, ]

# have to change Feasible and Nonfeasible for each scheduling solution

png(filename="C:/Users/long.mai/Dropbox/Weely_report/ML_extend/Histogram_Preshaping_high_resolution.png", units="in", width=8, height=6, pointsize=12, res=1200)

#par(mfrow=c(2,3)) # not allign
nf <- layout(matrix(c(1,1,2,2,3,3,0,4,4,5,5,0), 2, 6, byrow=TRUE), respect=FALSE)  # allign second row
# critical
h1 <- hist(Feasible$critical, plot = FALSE)
h2 <- hist(Nonfeasible$critical, plot = FALSE)
xlim = max(max(h1$breaks), h2$breaks)
ylim = max(max(h1$counts), h2$counts)
plot(h1, yaxp = c(0,2000,20),col=color_green, xlim = c(0,xlim), ylim = c(0,ylim),
     xlab='Number of critical flows', ylab='Frequency', main = 'Feasibility vs. # of critical flows')
plot(h2, yaxp = c(0,2000,20), col=color_red, add=TRUE)

# audio
h1 <- hist(Feasible$audio, plot = FALSE)
h2 <- hist(Nonfeasible$audio, plot = FALSE)
xlim = max(max(h1$breaks), h2$breaks)
ylim = max(max(h1$counts), h2$counts)
plot(h1, yaxp = c(0,2000,20),col=color_green, xlim = c(0,xlim), ylim = c(0,ylim),
     xlab='Number of audio flows', ylab='Frequency', main = 'Feasibility vs. # of audio flows')
plot(h2, yaxp = c(0,2000,20), col=color_red, add=TRUE)

# video
h1 <- hist(Feasible$video, plot = FALSE)
h2 <- hist(Nonfeasible$video, plot = FALSE)
xlim = max(max(h1$breaks), h2$breaks)
ylim = max(max(h1$counts), h2$counts)
plot(h1, yaxp = c(0,2000,20),col=color_green, xlim = c(0,xlim), ylim = c(0,ylim),
     xlab='Number of video flows', ylab='Frequency', main = 'Feasibility vs. # of video flows')
plot(h2, yaxp = c(0,2000,20), col=color_red, add=TRUE)

# maxLoad
h1 <- hist(Feasible$maxLoad, plot = FALSE)
h2 <- hist(Nonfeasible$maxLoad, plot = FALSE)
xlim = max(max(h1$breaks), h2$breaks)
ylim = max(max(h1$counts), h2$counts)
plot(h1, yaxp = c(0,2000,20),col=color_green, xlim = c(0,xlim), ylim = c(0,ylim),
     xlab='Max load of links', ylab='Frequency', main = 'Feasibility vs. max load')
plot(h2, yaxp = c(0,2000,20), col=color_red, add=TRUE)

# GiniIndex
h1 <- hist(Feasible$GiniIndex, plot = FALSE)
h2 <- hist(Nonfeasible$GiniIndex, plot = FALSE)
xlim = max(max(h1$breaks), h2$breaks)
ylim = max(max(h1$counts), h2$counts)
plot(h1, yaxp = c(0,2000,20),col=color_green, xlim = c(0,xlim), ylim = c(0,ylim),
     xlab='Gini index', ylab='Frequency', main = 'Feasibility vs. Gini index')
plot(h2, yaxp = c(0,2000,20), col=color_red, add=TRUE)
legend("topright", legend=c("Feasible", "Non-feasible"), fill = c(color_green, color_red))

dev.off()
################ End of plot histogram of each feature ###################################

