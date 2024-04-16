#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 12:19:51 2022
Modified on April 2024

@author: Mou
"""

import h5py
import os
import numpy as np
from sklearn import preprocessing
from sklearn import svm


def create_k_folds(n_splits, classes, stratified=False):
    """
    Creates a vector with as many elements as elements are in the dataset. Each
    element of such vector contains the number of the index of the fold in
    which that element should be. This assignation is made randomly.
    
    Parameters
        ----------
        n_splits: int
            Number of folds to generate.
        classes: numpy 1D array
            Vector that indicates the classes of the elements of the dataset.
        stratified: boolean
            Boolean variable which indicates whether the k-fold partition
            should be stratified (True) or not (False). Default value: False

    Returns
        -------
        indices_folds: numpy 1D array
            Vector (numpy 1D array) with the same length as the input vector y
            that contains the fold in which the corresponding element of the
            dataset should be.
            It means that, if the i-th position of the output vector is N, then
            the element X[i] of the dataset, whose class is y[i] will be in the
            N-th fold.
    """
    indices_folds = np.zeros(classes.shape, dtype=int)
    aux = np.zeros(classes.shape, dtype=int)
    if not stratified:
        # ====================== YOUR CODE HERE ======================
        shuffled_indices = np.random.permutation(len(classes))
        
        fold_size = len(classes) // n_splits
        for i in range(n_splits):
            start_idx = i * fold_size
            end_idx = start_idx + fold_size if i < n_splits - 1 else len(classes)
            indices_folds[shuffled_indices[start_idx:end_idx]] = i
        # ============================================================
    else:
        # ====================== YOUR CODE HERE ======================
        lista = []
        countsArr = []
        result = []
        differentClasses = len(np.unique(classes))
        print("different clases:", differentClasses)
        print("classes:" , classes)
        for i in range(differentClasses):
            count = len(np.nonzero(classes == i)[0]) 
            #print("count:", count)
            vec = np.arange(count)
            #print("vec:", vec)
            np.random.shuffle(vec)
            #print("vec:", vec)
            #print("lista:", lista)
            lista.append(vec % n_splits)
            #print("lista vec:", lista)
            #print("counts arr:", countsArr)
            countsArr.append(0)
            #print("counts app:", countsArr)
        for c in classes:
            cl = int(c)
            # =============================================================================
            #             Aquí se asigna un pliegue a la muestra actual. La lista result
            #             contendrá los índices de los pliegues asignados a cada muestra. Para la muestra actual, se obtiene el 
            #             pliegue correspondiente de la lista lista usando lista[cl]. countsArr[cl] se utiliza para obtener el 
            #             índice del pliegue en la lista lista que aún no ha sido asignado a ninguna muestra de la clase cl. Se 
            #             agrega este índice a la lista result como el pliegue asignado a la muestra actual.
            # =============================================================================
            result.append(lista[cl][countsArr[cl]])
            countsArr[cl] += 1
        
        indices_folds = np.array(result)
        # ============================================================

    return indices_folds


# %%
# -------------
# MAIN PROGRAM
# -------------
if __name__ == "__main__":

    np.random.seed(424)
    dir_data = "Data"
    data_path = os.path.join(dir_data, "mammographic_data.h5")
    
# %%
# -------------
# PRELIMINARY: LOAD DATASET
# -------------

    # Import data from the csv using pandas
    '''
    mammographic_data_df = pd.read_csv(data_path)
    y_df = mammographic_data_df[['Class']].copy()
    X_df = mammographic_data_df.copy()
    X_df = X_df.drop('Class', axis=1)

    X = X_df.to_numpy()
    y = y_df.to_numpy().flatten()

    # Import data from the h5 file (IN CASE IMPORTING FROM THE CSV DOES NOT
    # WORK PROPERLY)
    '''
    # import features and labels
    h5f_data = h5py.File(data_path, 'r')

    features_ds = h5f_data['data']
    labels_ds = h5f_data['labels']

    X = np.array(features_ds)
    y = np.array(labels_ds).flatten()

    h5f_data.close()

# %%
# -------------
# PART 1: CREATE K FOLDS AND CHECK THE PROPORTIONS
# -------------
    K = 10  # number of folds

    # Generate the indices of the folds by calling create_k_folds
    # ====================== YOUR CODE HERE ======================
    idx_folds = create_k_folds(K, y, True)
    # ============================================================
     
    proportion_class_0 = np.sum(y == 0) / y.size
    proportion_class_1 = 1 - proportion_class_0
    print("**********************************************************")
    print("****** CHECK THE CLASS PROPORTIONS WITHIN THE FOLDS ******")
    print("**********************************************************")
    print("\n")
    print("The distribution of the complete dataset is:")
    print("- {:.2f} % elements of class 0".format(
        100 * proportion_class_0))
    print("- {:.2f} % elements of class 1".format(
        100 * proportion_class_1))
    print("\n")
    print("The distribution of the elements within each fold is:")

    for i in range(K):
        # Obtain the indices of the test set elements (i.e., those in fold i)
        test_index = np.nonzero(idx_folds == i)[0]
        # Obtain the indices of the test set elements (i.e., those in the other
        # folds)
        train_index = np.nonzero(idx_folds != i)[0]

        prop_class_0_train = np.sum(y[train_index] == 0) / train_index.size
        prop_class_1_train = 1 - prop_class_0_train
        prop_class_0_test = np.sum(y[test_index] == 0) / test_index.size
        prop_class_1_test = 1 - prop_class_0_test
        print("* FOLD {}:".format(i+1))
        print("  - TRAIN: {:.2f} % elements of class 0;  {:.2f} % elements of class 1".format(
              100 * prop_class_0_train, 100 * prop_class_1_train))
        print("  - TEST: {:.2f} % elements of class 0;  {:.2f} % elements of class 1".format(
              100 * prop_class_0_test, 100 * prop_class_1_test))

# %%
# -------------
# PART 2: CROSS VALIDATION WITH SVM
# -------------

    # Parameters for SVM
    C_value = 1
    kernel_type = "rbf" # You should try different kernels. Read the documentation

    # Initialization of the vectors to store the accuracies and Fscores
    # of each fold
    accuracies = np.zeros(shape=(K,))
    Fscores = np.zeros(shape=(K,))

    # Cross-validation iterative process
    for i in range(K):
        # Use the indices of the test and train set elements of the i-th fold
        # to extract the train and test subsets of this fold.
        # ====================== YOUR CODE HERE ======================
        X_train_fold = X[np.nonzero(idx_folds != i)]
        y_train_fold = y[np.nonzero(idx_folds != i)]
        X_test_fold = X[np.nonzero(idx_folds == i)]
        y_test_fold = y[np.nonzero(idx_folds == i)]
        # ============================================================

        # Standardize data of this fold
        scaler = preprocessing.StandardScaler()
        scaler.fit(X_train_fold)
        X_train_fold = scaler.transform(X_train_fold)
        X_test_fold = scaler.transform(X_test_fold)

        # Instantiate the SVM with the defined kernel type and C value, train
        # it and use it to classify. Use the train and test sets of the current
        # iteration. 
        # ====================== YOUR CODE HERE ======================
        # Instantiate
        svc = svm.SVC(kernel=kernel_type, C=1, gamma=5.0)
        # Train
        svc.fit(X_train_fold, y_train_fold)
        # Classify test set
        y_test_assig_fold = svc.predict(X_test_fold)
        # ============================================================

        # Compute the accuracy and f-score of the test set in this fold and
        # store them in the vectors accuracies and Fscores, respectively
        # ====================== YOUR CODE HERE ======================
        accuracy_fold = np.count_nonzero(y_test_assig_fold == y_test_fold) / len(y_test_fold)
        precision = np.count_nonzero((y_test_fold == 1) & (y_test_assig_fold == 1)) / np.count_nonzero(y_test_assig_fold==1)
        recall = np.count_nonzero((y_test_assig_fold == 1) & (y_test_fold==1)) / np.count_nonzero(y_test_fold==1)
        Fscore_fold = 2*precision*recall/(precision+recall)
        
        accuracies[i] = accuracy_fold
        Fscores[i] = Fscore_fold
        # ============================================================


# %%
# -------------
# PART 3: SHOW FINAL RESULTS
# -------------

    print("\n\n")
    print('***********************************************')
    print('******* RESULTS OF THE CROSS VALIDATION *******')
    print('***********************************************')
    print('\n')

    for i in range(K):
        print("FOLD {}:".format(i+1))
        print("    Accuracy = {:4.3f}".format(accuracies[i]))
        print("    Fscore = {:5.3f}".format(Fscores[i]))

    # ====================== YOUR CODE HERE ======================
    # Calculate mean and std of the accuracies and F1-scores
    mean_accuracy = np.mean(accuracies)
    std_accuracy = np.std(accuracies)
    mean_fscore = np.mean(Fscores)
    std_fscore = np.std(Fscores)
    # ============================================================

    print("\n")
    print("AVERAGE ACCURACY = {:4.3f}; STD ACCURACY = {:4.3f}".format(
        mean_accuracy, std_accuracy))
    print("AVERAGE FSCORE = {:4.3f}; STD FSCORE = {:4.3f}".format(
        mean_fscore, std_fscore))
    print("\n")
    print('***********************************************')
    print('***********************************************')
    print('***********************************************')
    print("\n\n\n")
