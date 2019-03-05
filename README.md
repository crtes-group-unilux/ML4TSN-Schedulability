# ML4TSN-Schedulability
Code and data of the ML experiments of the technical report "Using Machine Learning to speed up the Design Space Exploration of Ethernet TSN networks" available at url: http://orbilu.uni.lu/bitstream/10993/38604/1/feasibility-with-ml.pdf


# Explanation of R code #

knn.R is code to train k-NN. Include code for 5-fold evaluation and code for testing with variation from 10% - 90%

kmeans.R is code to train K-means. Include code for evaluation without variaion, and code for testing with variation from 10 - 90%. NOTE: training K-means takes more time than k-NN.



# Explanation of data set #


## Data set in TrainingDataKNN.txt is training set of k-NN ##

| No. |			Column name		   |				Explanation |
|:----|:---------------------|:-------------------|
| 1		|	totalIndividualFlows |			# of flows in the configurations (Note: remove totalIndividualFlows = 10) |
| 2		|	Critical						 | # of critical flows |	
| 3		|	Audio							   | # of audio flows |
| 4		|	Video							   | # of video flow |
| 5		|	maxLoad							 | Maximum load of all links |
| 10	|	FIFO							   | # of non-feasible flows with FIFO scheduling |
| 12	|	ManualClassification | # of non-feasible flows with Manual scheduling |
| 14	|	ConcisePriorities8classes |	# of non-feasible flows with CP8 scheduling |
| 197-205 |	LoadOfEthernetLink_(0-8) | load of network links |
| 208	|	solutionsOfPreshaping	|	# of feasible solutions found by Preshaping scheduling |
| 212 | GiniIndex |	Imbalance of load in network links |

Note: GiniIndex is derived from load of network links (columns 197 - 205)

Column (2,3,4,5,212) are features. Column (10,12,14,208) are labels, that are feasible / non-feasible

In column (10,12,14): if there is at least 1 non-feasible flows in the configuration -> the configuration is labelled non-feasible.

In column (205): if there is at least 1 feasible solution founded by Preshaping -> The configuration is labelled feasible.



## Data set in TrainingDataKmeans1.txt and TrainingDataKmeans2.txt is training set of K-means ##

The training set of K-means is divided into 2 files since it is large.

Information of the feature columns (2,3,4,5,212) in the training set of K-means is the same as in the training set of k-NN

IMPORTANCE: There is no label column in the training set of K-means.



## Data set in TestingDataVariation0percent.txt is testing set of k-NN / K-means ##

Information of position and name of columns in testing set is the same as in training set of k-NN, exception the GiniIndex (column 217).

Note: Information of position and name of columns in testing set are the same for all variation (0% - 90%).
