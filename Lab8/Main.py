#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 20:42:04 2024

@author: moasin
"""

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
import sklearn as sk

import matplotlib.pyplot as plt

# import to KNN
from sklearn.neighbors import KNeighborsClassifier
# Import to LogisticRegression
from sklearn.linear_model import LogisticRegression
# Import to SVM
from sklearn.svm import SVC
# Import to DecisionTree
from sklearn.tree import DecisionTreeClassifier


def knnClassFunction(X_train, X_test, y_train, y_test, k):
    knnClass = KNeighborsClassifier(n_neighbors=k)
    knnClass.fit(X_train, y_train)
    predKnn = knnClass.predict(X_test)

    scoreKnn = accuracy_score(y_test, predKnn)

    print("Score knn axcuuracy: ", scoreKnn)
    confMatrixKnn = confusion_matrix(y_true=y_test, y_pred=predKnn)
    disp = ConfusionMatrixDisplay(confusion_matrix=confMatrixKnn)
    disp.plot()
    title = 'Confusion Matrix KNN\n Accuracy score: ' + str(scoreKnn)
    plt.title(title, fontsize=14)
    plt.show()


def logisticRegresionClassFunction(X_train, X_test, y_train, y_test):
    logReg = LogisticRegression()
    logReg.fit(X_train, y_train)
    predLogClass = logReg.predict(X_test)

    scoreLog = accuracy_score(y_test, predLogClass)
    print("Score logistic axcuuracy: ", scoreLog)
    confMatrixLogRegression = confusion_matrix(
        y_true=y_test, y_pred=predLogClass)
    disp = ConfusionMatrixDisplay(confusion_matrix=confMatrixLogRegression)
    disp.plot()
    title = 'Confusion Matrix Logistic Regresion\n Accuracy score: ' + \
        str(scoreLog)
    plt.title(title, fontsize=14)

    plt.show()


def svmClassFunction():
    kernel = "rbf"
    pass


def decisionTreeClassFunction(X_train, X_test, y_train, y_test):
    decTree = DecisionTreeClassifier()
    decTree.fit(X_train, y_train)
    predDecTree = decTree.predict(X_test)

    scoreDecTree = accuracy_score(y_test, predDecTree)
    print("Score decision tree axcuuracy: ", scoreDecTree)

    confMatrixDecisionTree = confusion_matrix(
        y_true=y_test, y_pred=predDecTree)
    disp = ConfusionMatrixDisplay(confusion_matrix=confMatrixDecisionTree)
    disp.plot()
    title = 'Confusion Matrix Decision Tree\n Accuracy score: ' + \
        str(scoreDecTree)
    plt.title(title, fontsize=14)
    plt.show()


def main():

    spermData = pd.read_csv("Data/data_sperm_WCF4.csv")
    # print(df.head)
    # columnas excepto las de las clases
    X = spermData.drop(columns=['Class'])
    # columna de las clases conocidas
    y = spermData["Class"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    print("#############--Xtrain---##################")
    print(X_train)
    print("################################################")

    print("#############--Xtest---##################")
    print(X_test)
    print("################################################")

    print("#############--Ytrain---##################")
    print(y_train)
    print("################################################")

    print("#############--Ytest---##################")
    print(y_test)
    print("################################################")

    print("################################################")
    knnClassFunction(X_train, X_test, y_train, y_test, 5)
    print("################################################")

    
    print("################################################")
    logisticRegresionClassFunction(X_train, X_test, y_train, y_test)
    print("################################################")


    print("################################################")
    decisionTreeClassFunction(X_train, X_test, y_train, y_test)
    print("################################################")


if __name__ == "__main__":
    main()
