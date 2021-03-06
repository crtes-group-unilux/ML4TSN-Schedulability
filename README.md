# ML4TSN-Schedulability
Code and data of the ML experiments of
- the technical report "Using Machine Learning to speed up the Design Space Exploration of Ethernet TSN networks" available at url: http://orbilu.uni.lu/bitstream/10993/38604/1/feasibility-with-ml.pdf
- the paper: "A Hybrid Machine Learning and Schedulability Method for the Verification of TSN Networks", T.L. Mai, N. Navet, J. Migge, to appear at the 15th IEEE International Workshop on Factory Communication System (WFCS 2019), Sundsvall, Sweden, May 27-29, 2019. Authors' preprint available at http://orbilu.uni.lu/bitstream/10993/38990/1/hybrid-verification-TSN.pdf


# Explanation of R code #

- knn.R is code to train k-NN. It includes code for 5-fold evaluation and code for testing with variation of the payload from 10% - 90%
- kmeans.R is code to train K-means. It includes code for evaluation without payload variation, and code for testing with variation from 10 - 90%. NOTE: training K-means takes more time than k-NN.
- hybrid.R is the code for the hybrid approach presented in the WFCS'2019 paper.


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



# Explanation of R code and data sets of the new topology - submitted to RTNS 2019 #

knn_newTopo.R is the R code for new topology that is used in the paper submitted to RTNS 2019.

Note: The columns in the training sets and testing sets for new topology are changed as follow.

| No. |			Column name		   |				Explanation |
|:----|:---------------------|:-------------------|
| 1		|	totalIndividualFlows | # of flows in the configurations (Note: remove totalIndividualFlows = 10) |
| 2		|	Critical						 | # of critical flows |	
| 3		|	Audio							   | # of audio flows |
| 4		|	Video							   | # of video flow |
| 5		|	maxLoad							 | Maximum load of all links |
| 10	|	FIFO							   | # of non-feasible flows with FIFO scheduling |
| 12	|	ManualClassification | # of non-feasible flows with Manual scheduling |
| 14	|	ConcisePriorities8classes |	# of non-feasible flows with CP8 scheduling |
| 217-252 |	LoadOfEthernetLink_(0-18)_(1_2) | load of network links |
| 255	|	solutionsOfPreshaping	|	# of feasible solutions found by Preshaping scheduling |
| 264 | GiniIndex |	Imbalance of load in network links |

Note: GiniIndex is derived from load of network links (columns 217 - 252)

Column (2,3,4,5,264) are features. Column (10,12,14,255) are labels, that are feasible / non-feasible

In column (10,12,14): if there is at least 1 non-feasible flows in the configuration -> the configuration is labelled non-feasible.

In column (255): if there is at least 1 feasible solution founded by Preshaping -> The configuration is labelled feasible.
