#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Mouhcine El Oualidi Charchmi
@version: 1.0
@note: The prints used in the application development have been retained for 
runtime viewing. Additionally, the code necessary for displaying the confusion
matrices has been commented out in each of the custom-defined functions. 
This decision was made because displaying the confusion matrices slightly 
impacts performance, leading to potentially lengthy execution times (2-7 seconds).
"""


import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# import to KNN
from sklearn.neighbors import KNeighborsClassifier
# Import to LogisticRegression
from sklearn.linear_model import LogisticRegression
# Import to SVM
from sklearn.svm import SVC
# Import to DecisionTree
from sklearn.tree import DecisionTreeClassifier
# import to normalize data
from sklearn.preprocessing import StandardScaler
# import for tables
from tabulate import tabulate


"""
    Function to evaluate K-Nearest Neighbors classifier.
    :param X_train: Training features
    :param X_test: Test features
    :param y_train: Training labels
    :param y_test: Test labels
    :param k: Number of neighbors
    :param normalized: Whether the data is normalized or not (default=False)
    :return: Accuracy score and F1 score
"""


def knnClassFunction(X_train, X_test, y_train, y_test, k, normalized=False):
    knnClass = KNeighborsClassifier(n_neighbors=k)
    knnClass.fit(X_train, y_train)
    predKnn = knnClass.predict(X_test)

    scoreKnn = accuracy_score(y_test, predKnn)
    # print("Score knn axcuuracy: ", scoreKnn)

    f1ScoreKnn = f1_score(y_test, predKnn)
    # print("Score knn f1: ", f1ScoreKnn)

    # confMatrixKnn = confusion_matrix(y_true=y_test, y_pred=predKnn)
    # disp = ConfusionMatrixDisplay(confusion_matrix=confMatrixKnn)
    # disp.plot()
    # if(normalized):
    #     title = 'Confusion Matrix KNN with k= ' + \
    #         str(k) + '\n Accuracy score: ' + \
    #         str(scoreKnn) + '\n With Normalized Data'
    # else:
    #     title = 'Confusion Matrix KNN with k= ' + \
    #         str(k) + '\n Accuracy score: ' + str(scoreKnn)
    # plt.title(title, fontsize=14)
    # plt.show()
    return scoreKnn, f1ScoreKnn


"""
    Function to evaluate Logistic Regression classifier.
    :param X_train: Training features
    :param X_test: Test features
    :param y_train: Training labels
    :param y_test: Test labels
    :param k: Number of neighbors
    :param normalized: Whether the data is normalized or not (default=False)
    :return: Accuracy score and F1 score
"""


def logisticRegresionClassFunction(X_train, X_test, y_train, y_test, normalized=False):
    logReg = LogisticRegression()
    logReg.fit(X_train, y_train)
    predLogClass = logReg.predict(X_test)

    scoreLog = accuracy_score(y_test, predLogClass)
    # print("Score logistic axcuuracy: ", scoreLog)

    f1ScoreLogRegre = f1_score(y_test, predLogClass)
    # print("Score logistic f1: ", f1ScoreLogRegre)

    # confMatrixLogRegression = confusion_matrix(
    #     y_true=y_test, y_pred=predLogClass)
    # disp = ConfusionMatrixDisplay(confusion_matrix=confMatrixLogRegression)
    # disp.plot()

    # if(normalized):
    #     title = 'Confusion Matrix Logistic Regresion\n Accuracy score: ' + str(scoreLog) + '\nwith normalized data'
    # else:
    #     title = 'Confusion Matrix Logistic Regresion\n Accuracy score: ' + str(scoreLog)

    # plt.title(title, fontsize=14)

    # plt.show()
    return scoreLog, f1ScoreLogRegre


"""
    Function to evaluate Support Vector Machine classifier(SVM).
    :param X_train: Training features
    :param X_test: Test features
    :param y_train: Training labels
    :param y_test: Test labels
    :param kernelType: Type of kernel
    :param C: Regularization parameter
    :param normalized: Whether the data is normalized or not (default=False)
    :return: Accuracy score and F1 score
"""


def svmClassFunction(X_train, X_test, y_train, y_test, kernelType, C, normalized=False):
    svm = SVC(C=C, kernel=kernelType)

    svm.fit(X_train, y_train)

    predsSvm = svm.predict(X_test)

    scoreSVM = accuracy_score(y_test, predsSvm)
    # print("Score svm axcuuracy: ", scoreSVM)

    f1ScoreSVM = f1_score(y_test, predsSvm)
    # print("Score SVM f1: ", f1ScoreSVM)

    # confMatrixSVM = confusion_matrix(
    #     y_true=y_test, y_pred=predsSvm)
    # disp = ConfusionMatrixDisplay(confusion_matrix=confMatrixSVM)
    # disp.plot()

    # if(normalized):
    #     title = 'Confusion Matrix SVM Kernel=' + \
    #         str(kernelType) + ' C=5\n Accuracy score: ' + \
    #         str(scoreSVM) + '\n With Normalized Data'
    # else:
    #     title = 'Confusion Matrix SVM Kernel=' + \
    #         str(kernelType) + ' C=5\n Accuracy score: ' + str(scoreSVM)

    # plt.title(title, fontsize=14)
    # plt.show()
    return scoreSVM, f1ScoreSVM


"""
    Function to evaluate Decision Tree classifier.
    :param X_train: Training features
    :param X_test: Test features
    :param y_train: Training labels
    :param y_test: Test labels
    :param normalized: Whether the data is normalized or not (default=False)
    :return: Accuracy score and F1 score
"""


def decisionTreeClassFunction(X_train, X_test, y_train, y_test, normalized=False):
    decTree = DecisionTreeClassifier()
    decTree.fit(X_train, y_train)
    predDecTree = decTree.predict(X_test)

    scoreDecTree = accuracy_score(y_test, predDecTree)
    # print("Score decision tree axcuuracy: ", scoreDecTree)

    f1DecTree = f1_score(y_test, predDecTree)
    # print("Score Decision tree f1: ", f1DecTree)

    # confMatrixDecisionTree = confusion_matrix(
    #     y_true=y_test, y_pred=predDecTree)
    # disp = ConfusionMatrixDisplay(confusion_matrix=confMatrixDecisionTree)
    # disp.plot()

    # if(normalized):
    #     title = 'Confusion Matrix Decision Tree\n Accuracy score: ' + str(scoreDecTree) + '\n with Normalized data'
    # else:
    #     title = 'Confusion Matrix Decision Tree\n Accuracy score: ' + str(scoreDecTree)

    # plt.title(title, fontsize=14)
    # plt.show()a
    return scoreDecTree, f1DecTree


"""
Main function to perform classifier evaluations.

"""


def main():

    spermData = pd.read_csv("Data/data_sperm_WCF4.csv")
    # columns except for the class ones
    X = spermData.drop(columns=['Class'])
    # column of known classes
    y = spermData["Class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42)

    # normalized data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("Evaluating classifiers...")
    # print("################--KNN Classifier--################")
    # Evaluate KNN with k=5
    # print("Evaluating KNN with k=5...")
    acurracyScoreKnn5, f1Scoreknn5 = knnClassFunction(
        X_train, X_test, y_train, y_test, k=5)
    # print("------------------------------------------------")
    # Evaluate KNN with k=10
    # print("Evaluating KNN with k=10...")
    acurracyScoreKnn10, f1Scoreknn10 = knnClassFunction(
        X_train, X_test, y_train, y_test, k=10)
    # print("------------------------------------------------")
    # Evaluate KNN with k=5 and normalized
    # print("Evaluating KNN with k=5 & normalized...")
    acurracyScoreKnn5Norm, f1Scoreknn5Norm = knnClassFunction(
        X_train_scaled, X_test_scaled, y_train, y_test, k=5, normalized=True)
    # print("------------------------------------------------")
    # Evaluate KNN with k=10 and normalized
    # print("Evaluating KNN with k=10 & normalized...")
    acurracyScoreKnn10Norm, f1Scoreknn10Norm = knnClassFunction(
        X_train_scaled, X_test_scaled, y_train, y_test, k=10, normalized=True)
    # print("##################################################")

    # print("################--SVM Classifier--################")
    # SVM with RBF kernel and C=5
    # print("Evaluating SVM with RBF kernel and C=5...")
    accuracyScoreSVMRBF, f1ScoreSVMRBF = svmClassFunction(
        X_train, X_test, y_train, y_test, kernelType='rbf', C=5)
    # print("------------------------------------------------")
    # SVM with linear kernel and C=5
    # print("Evaluating SVM with Linear kernel and C=5...")
    accuracyScoreSVMLinear, f1ScoreSVMLinear = svmClassFunction(
        X_train, X_test, y_train, y_test, kernelType='linear', C=5)
    # print("------------------------------------------------")
    # SVM with RBF kernel and C=5 with normalized data
    # print("Evaluating SVM with RBF kernel and C=5 with normalized data...")
    accuracyScoreSVMRBFNorm, f1ScoreSVMRBFNorm = svmClassFunction(
        X_train_scaled, X_test_scaled, y_train, y_test, kernelType='rbf', C=5, normalized=True)
    # print("------------------------------------------------")
    # SVM with linear kernel and C=5 with normalized data
    # print("Evaluating SVM with Linear kernel and C=5 with normalized data...")
    accuracyScoreSVMLinearNorm, f1ScoreSVMLinearNorm = svmClassFunction(
        X_train_scaled, X_test_scaled, y_train, y_test, kernelType='linear', C=5, normalized=True)
    # print("##################################################")

    # print("################--Logistic Regression Classifier--################")
    # SVM with RBF kernel and C=5
    # print("Evaluating Logistic regresion...")
    accuracyScoreLogRegres, f1ScoreLogRegres = logisticRegresionClassFunction(
        X_train, X_test, y_train, y_test)
    # print("------------------------------------------------")
    # print("Evaluating Logistic regresion with normalized data...")
    accuracyScoreLogRegresNorm, f1ScoreLogRegresNorm = logisticRegresionClassFunction(
        X_train_scaled, X_test_scaled, y_train, y_test, normalized=True)
    # print("##################################################")

    # print("################--Decision Tree Classifier--################")
    # print("Evaluating Decision Tree...")
    accuracyScoreDecTree, f1ScoreDecTree = decisionTreeClassFunction(
        X_train, X_test, y_train, y_test)
    # print("------------------------------------------------")
    # print("Evaluating Decision Tree with normalized data...")
    accuracyScoreDecTreeNorm, f1ScoreDecTreeNorm = decisionTreeClassFunction(
        X_train_scaled, X_test_scaled, y_train, y_test, normalized=True)
    # print("##################################################")

    # Define results as a dictionary for tabular printing
    results = {
        'KNN': {
            'K=5': {'Accuracy': acurracyScoreKnn5, 'F1 Score': f1Scoreknn5},
            'K=10': {'Accuracy': acurracyScoreKnn10, 'F1 Score': f1Scoreknn10},
            'Normalized data K=5': {'Accuracy': acurracyScoreKnn5Norm, 'F1 Score': f1Scoreknn5Norm},
            'Normalized data K=10': {'Accuracy': acurracyScoreKnn10Norm, 'F1 Score': f1Scoreknn10Norm}
        },
        'SVM': {

            'RBF (C=5)': {'Accuracy': accuracyScoreSVMRBF, 'F1 Score': f1ScoreSVMRBF},
            'Lineal (C=5)': {'Accuracy': accuracyScoreSVMLinear, 'F1 Score': f1ScoreSVMLinear},
            'Normalized data RBF (C=5)': {'Accuracy': accuracyScoreSVMRBFNorm, 'F1 Score': f1ScoreSVMRBFNorm},
            'Normalized data Lineal (C=5)': {'Accuracy': accuracyScoreSVMLinearNorm, 'F1 Score': f1ScoreSVMLinearNorm}
        },
        'Logistic Regression': {
            'Normalized data No': {'Accuracy': accuracyScoreLogRegres, 'F1 Score': f1ScoreLogRegres},
            'Normalized data Yes': {'Accuracy': accuracyScoreLogRegresNorm, 'F1 Score': f1ScoreLogRegresNorm}
        },
        'Decision Tree': {
            'Normalized data No': {'Accuracy': accuracyScoreDecTree, 'F1 Score': f1ScoreDecTree},
            'Normalized data Yes': {'Accuracy': accuracyScoreDecTreeNorm, 'F1 Score': f1ScoreDecTreeNorm}
        }
    }

    # Create a list of tuples for each row of the table
    table_rows = []
    for classifier, params in results.items():
        for param, metrics in params.items():
            accuracy = metrics['Accuracy']
            f1_score = metrics['F1 Score']
            table_rows.append((classifier, param, accuracy, f1_score))

   # Print the table using tabulate
    print(tabulate(table_rows, headers=[
          'CLASSIFIER', 'PARAMETERS', 'ACCURACY', 'F1Score'], tablefmt='grid'))
    
    # Find the algorithm with the best accuracy and F1 Score
    best_algorithm = max(results.items(), key=lambda x: max([y['Accuracy'] for y in x[1].values()]))
    
    # Get the name of the classifier and the parameters of the best algorithm
    classifier_name = best_algorithm[0]
    parameters = max(best_algorithm[1], key=lambda x: best_algorithm[1][x]['Accuracy'])
    
    # Get the accuracy and F1 Score data of the best algorithm
    accuracy = results[classifier_name][parameters]['Accuracy']
    f1_score = results[classifier_name][parameters]['F1 Score']
    
    # Create a list with the data of the best algorithm
    best_algorithm_row = [(classifier_name, parameters, accuracy, f1_score)]
    
    # Print the row of the best algorithm
    print("Best Classifier")
    print(tabulate(best_algorithm_row, headers=['CLASSIFIER', 'PARAMETERS', 'ACCURACY', 'F1Score'], tablefmt='grid'))

    
if __name__ == "__main__":
    main()
