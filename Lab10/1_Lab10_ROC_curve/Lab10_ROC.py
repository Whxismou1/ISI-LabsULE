#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 13:31:33 2022
Modified on Mon May 6 2024

@author: Whxismou
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


def get_ROC(y_test_probs, y_test, num_thresholds):
    """
    This function obtains the values of the 1-specificity (False Positive Rate
     - FPR) and the senstivity (True Positive Rate - TPR) that define the ROC
    curve, for a given number of decision thresholds. Besides, it computes the
    area under the calculated ROC curve. This function is only designed for
    binary classification problems.

    Parameters
    ----------
    y_test_probs : Numpy vector
        Vector that contains the probabilities of belonging to class 1 yielded
        by the classifier, for each sample in the test set.
    y_test : Numpy vector
        Vector that contains the real classes (0 or 1) of the samples in the
        test set.
    num_thresholds : Integer
        The number of decision thresholds to be used to calculate the
        sensitivity and 1-specificity values.

    Returns
    -------
    v_FPR : Numpy Vector
        Vector that contains the values of 1-specificity (i.e. FPR) for each
        decision threshold.
    v_TPR : Numpy Vector
        Vector that contains the values of sensitivity (i.e. TPR) for each
        decision threshold.
    """

    # Initialize output vectors
    v_FPR = np.zeros(num_thresholds)
    v_TPR = np.zeros(num_thresholds)

    # ====================== YOUR CODE HERE ======================
    # Define thresholds
    thresholds = np.linspace(0, 1, num_thresholds)

    # Compute TPR and FPR for each threshold
    for i, threshold in enumerate(thresholds):
        # Predict classes based on threshold
        y_pred = y_test_probs >= threshold
        # Compute confusion matrix
        conf_matrix = confusion_matrix(y_test, y_pred)
        # Extract TN, FP, FN, TP
        TN, FP, FN, TP = conf_matrix.ravel()
        # Compute TPR and FPR
        TPR = TP / (TP + FN)
        FPR = FP / (FP + TN)
        # Store values
        v_TPR[i] = TPR
        v_FPR[i] = FPR
    # ============================================================

    return v_FPR, v_TPR


# %%
# -------------
# MAIN PROGRAM
# -------------
if __name__ == "__main__":

    np.random.seed(524)

    dir_data = "Data"
    data_path = os.path.join('..', dir_data, "mammographic_data.csv")
    test_size = 0.3

# -------------
# PRELIMINARY: LOAD DATASET AND PARTITION TRAIN-TEST SETS (NO NEED TO CHANGE
# ANYTHING)
# -------------

    # import features and labels
    # Load the data
    data_df = pd.read_csv(data_path)
    y = data_df['Class'].to_numpy().ravel()
    X = data_df.copy().drop('Class', axis=1).to_numpy()

    # SPLIT DATA INTO TRAINING AND TEST SETS
    # ====================== YOUR CODE HERE ======================
    # DO NOT use the parameter random_state
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size)
    # ============================================================

    # NORMALIZE DATA (normalization by standardization of the features)
    # ====================== YOUR CODE HERE ======================
    feature_means = np.mean(X_train, axis=0)
    feature_std = np.std(X_train, axis=0)

    X_train_normalized = (X_train - feature_means) / feature_std
    X_test_normalized = (X_test - feature_means) / feature_std
    # ============================================================

    # Uncomment if you want to check that the mean and std are close to 0 and 1
    # respectively
    # print("Mean of the training set: {}".format(X_train.mean(axis=0)))
    # print("Std of the training set: {}".format(X_train.std(axis=0)))
    # print("Mean of the test set: {}".format(X_test.mean(axis=0)))
    # print("Std of the test set: {}".format(X_test.std(axis=0)))

# %%
# -------------
# PART 1: CLASSIFICATION WITH SCIKIT-LEARN'S LOGISTIC REGRESSION
# -------------

    # Instance of the logistic regression model
    logit_model = LogisticRegression()

    # Train the model
    logit_model.fit(X_train, y_train)

    # Predict the probabilities of belonging to class 1 of the test samples
    y_test_hat = logit_model.predict_proba(X_test)[:, 1]

    # Predict the classes of the test set samples
    y_test_assig = logit_model.predict(X_test)

    # Test of the probabilities. If everything is right, it should print True
    # y_test_assig_proba = y_test_hat >= 0.5
    # print((y_test_assig == y_test_assig_proba).all())

    # Display confusion matrix when the decision threshold is 0.5
    confm = confusion_matrix(y_true=y_test, y_pred=y_test_assig)
    disp = ConfusionMatrixDisplay(confusion_matrix=confm)
    disp.plot()
    plt.title("Confusion Matrix for the logistic regression classifier",
              fontsize=14)
    plt.show()

# %%
# -------------
# PART 2: COMPUTATION AND PLOT OF THE ROC CURVE
# -------------

    # Calling the function to build the ROC curve
    # ====================== YOUR CODE HERE ======================
    v_FPR, v_TPR = get_ROC(y_test_hat, y_test, 100)
    # ============================================================

    # AREA UNDER THE ROC CURVE
    # Integration of the ROC curve using the trapezoidal rule
    # ====================== YOUR CODE HERE ======================
    AUC = np.trapz(v_TPR, v_FPR)
    # ============================================================

    # Plot of the curve
    plt.figure(2)
    plt.plot(v_FPR, v_TPR, 'b-', label="ROC of classifier")
    plt.plot([0, 1], [0, 1], 'r--', label="Random classification")
    plt.legend(loc='lower right', shadow=True)
    plt.xlabel("FPR (1-specificity)")
    plt.ylabel("TPR (sensitivity)")
    plt.title("ROC curve (AUC={:.3f})".format(AUC), fontsize=14)
    plt.xlim([-0.001, 1.001])
    plt.ylim([-0.001, 1.001])
    plt.show()
