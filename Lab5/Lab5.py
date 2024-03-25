#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 14 16:12:41 2022
Modified on Mon Mar 13 2023

@author: CHANGE THE NAME
"""

# Import whatever else you need to import
from scipy.stats import mode
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def classify_kNN(X_train, y_train, X_test, k):
    """
    This function implements the kNN classification algorithm with the
    eucludean distance

    Parameters
    ----------
    X_train : ndarray
        Matrix (n_Train x m), where n_Train is the number of training elements
        and m is the number of features (the length of the feature vector).
    y_train : ndarray
        The classes of the elements in the training set. It is a
        vector of length n_Train with the number of the class.
    X_test : ndarray
        matrix (n_t x m), where n_t is the number of elements in the test set.
    k : int
        Number of the nearest neighbours to consider in order to make an
        assignation.

    Returns
    -------
    y_test_assig : ndarray
        A vector with length n_t, with the classess assigned by the algorithm
        to the elements in the training set.
    """

    # num_elements_train = X_train.shape[0]
    num_elements_test = X_test.shape[0]
    
    # Allocate space for the output vector of assignations
    y_test_assig = np.empty(shape=(num_elements_test, 1), dtype=int)

    # For each element in the test set...
    for i in range(num_elements_test):
        """
        1 - Compute the Euclidean distance of the i-th test element to all the
        training elements
        """
        # ====================== YOUR CODE HERE ======================
        distances = np.linalg.norm(X_train - X_test[i], axis=1)
        # ============================================================

        """
        2 - Order distances in ascending order and use the indices of the
        ordering
        """
        # ====================== YOUR CODE HERE ======================
        kNearestIndices = np.argsort(distances)[:k]
        # ============================================================

        """
        3 - Take the k first classes of the training set
        """
        # ====================== YOUR CODE HERE ======================
        kNearestClasses = y_train[kNearestIndices]
        # ============================================================

        """
        4 - Assign to the i-th element the most frequent class
        """
        # ====================== YOUR CODE HERE ======================
        y_test_assig[i] = mode(kNearestClasses)[0][0]
        # ============================================================

    return y_test_assig


# -------------
# MAIN PROGRAM
# -------------
if __name__ == "__main__":

    # PART 1: LOAD DATASET AND TRAIN-TEST PARTITION

    # Load csv with data into a pandas dataframe
    """
    Each row in this dataframe contains a feature vector, which represents an
    email.
    Each column represents a variable, EXCEPT LAST COLUMN, which represents
    the true class of the corresponding element (i.e. row): 1 means "spam",
    and 0 means "not spam"
    """
    dir_data = "Data"
    spam_df = pd.read_csv(os.path.join(dir_data, "spambase_data.csv"))
    y_df = spam_df[['Class']].copy()
    X_df = spam_df.copy()
    X_df = X_df.drop('Class', axis=1)

    # Convert dataframe to numpy array
    X = X_df.to_numpy()
    y = y_df.to_numpy()

    """
    Number of elements of the dataset and number of variables of each feature
    vector that represents each spam email
    """
    num_elements, num_variables = X.shape

    """
    Parameter that indicates the proportion of elements that the test set will
    have
    """
    proportion_test = 0.3

    """
    In the following section the partition of the dataset into training and
    test sets is done. Look the results produced by each line of code to
    understand what it does, using the debugger if necessary.
    
    Then, write a brief explanation for each line with comments.
    """
    # ============================================
    #Obtenemos el numero de elementos a entrenar
    num_elements_train = int(num_elements * (1 - proportion_test))

    #Se obtiene un indice de permutacion para tener elementos aleatorios
    inds_permutation = np.random.permutation(num_elements)

    # Se seleccionan los primeros num_elements_train índices de la permutación 
    #generada anteriormente para el conjunto de entrenamiento. Esto garantiza 
    #que se seleccionen aleatoriamente los elementos de entrenamiento.
    inds_train = inds_permutation[:num_elements_train]
    
    #se seleccionan los indices restantes para el cnjuton de prueba
    inds_test = inds_permutation[num_elements_train:]

    X_train = X[inds_train, :]
    y_train = y[inds_train]

    X_test = X[inds_test, :]
    y_test = y[inds_test]
    # ============================================
    # ***********************************************************************
    # ***********************************************************************
    # PART 2: K-NEAREST NEIGHBOURS ALGORITHM

    k = 3
    """
    The function classify_kNN implements the kNN algorithm. Go to it and
    complete the code
    """
    y_test_assig = classify_kNN(X_train, y_train, X_test, k)

    # ***********************************************************************
    # ***********************************************************************
    # PART 3: ASSESSMENT OF CLASSIFIER'S PERFORMANCE

    # Show confusion matrix
    confusion_matrix_kNN = confusion_matrix(y_true=y_test, y_pred=y_test_assig)

    # If you want to print the confusion matrix using matplotlib
    """
    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    ax.matshow(confusion_matrix_kNN, cmap=plt.cm.Blues, alpha=0.3)
    for i in range(confusion_matrix_kNN.shape[0]):
        for j in range(confusion_matrix_kNN.shape[1]):
            ax.text(x=j, y=i, s=confusion_matrix_kNN[i, j], va='center',
                    ha='center', size='xx-large')

    plt.xlabel('Predictions', fontsize=18)
    plt.ylabel('Actuals', fontsize=18)
    plt.title('Confusion Matrix', fontsize=18)
    plt.show()
    """
    disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix_kNN)
    disp.plot()
    plt.title('Confusion Matrix', fontsize=14)
    plt.show()

    # Accuracy: Proportion of elements well classified amogst all elements
    # ====================== YOUR CODE HERE ======================
    accuracy = (confusion_matrix_kNN[0, 0] + confusion_matrix_kNN[1, 1]) / np.sum(confusion_matrix_kNN)
    # ============================================================
    print('Accuracy: {:.4f}'.format(accuracy))

    # Sensitivity: Proportion of well classified elements amongst the real
    # positives
    # ====================== YOUR CODE HERE ======================
    sensitivity =  confusion_matrix_kNN[1, 1] / (confusion_matrix_kNN[1, 0] + confusion_matrix_kNN[1, 1])
    # ============================================================
    print('Sensitivity (TPR): {:.4f}'.format(sensitivity))

    # Specificity: Proportion of well classified elements amongst the real
    # NEGATIVES
    # ====================== YOUR CODE HERE ======================
    specificity = confusion_matrix_kNN[0, 0] / (confusion_matrix_kNN[0, 1] + confusion_matrix_kNN[0, 0])
    # ============================================================
    print('Specificity (TNR): {:.4f}'.format(specificity))
