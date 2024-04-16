#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 20:42:04 2024

@author: moasin
"""

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
# import to KNN
from sklearn.neighbors import KNeighborsClassifier
# Import to LogisticRegression
from sklearn.linear_model import LogisticRegression
# Import to SVM
from sklearn.svm import SVC
# Import to DecisionTree
from sklearn.tree import DecisionTreeClassifier


def knnClass():
    k = 5
    pass


def logisticClass():
    pass


def svmClass():
    kernel = "rbf"
    pass


def decisionTreeClass(X_train, X_test, y_train, y_test):
    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    score = accuracy_score(y_test, pred)
    print(score)


def main():

    spermData = pd.read_csv("Data/data_sperm_WCF4.csv")
    # print(df.head)
    # columnas excepto las de las clases
    X = spermData.drop(columns=['Class'])
    # columna de las clases conocidas
    y = spermData["Class"]
    print(X)
    print("---------------")
    print(y)

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
    decisionTreeClass(X_train, X_test, y_train, y_test)
    print("################################################")



if __name__ == "__main__":
    main()
