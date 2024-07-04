# LRSVM
This is the code for LRSVM in paper "Support Vector Machine with Discriminative Low-Rank Embedding"

|algorithm|code|
|-|-|
|LRSVM|LRSVM_1_1_Recognition.m|
|LRSVMOA|LRSVM_1_2_Recognition.m|
|LRSVMOB|LRSVM_1_3_Recognition.m|

Input:

    train, test: the training set and testing set, each row is a data point
    train_labels, test_labels: labels, one-hot form
    Lambda: hyperparameter of low-rank regression term
    C: hyperparameter of SVM
    s: number of rank
    
Output:

    train_error_rate: error of the training set
    test_error_rate: error of the testing set
    Fval： Value of objective function
