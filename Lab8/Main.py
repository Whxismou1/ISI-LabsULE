
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
#import to normalize data
from sklearn.preprocessing import StandardScaler

from tabulate import tabulate


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

def main():

    spermData = pd.read_csv("Data/data_sperm_WCF4.csv")
    # print(df.head)
    # columnas excepto las de las clases
    X = spermData.drop(columns=['Class'])
    # columna de las clases conocidas
    y = spermData["Class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42)

    # datos normalizados
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("Loading...")
    # print("################--KNN Classifier--################")
    # Evaluar KNN con k=5
    # print("Evaluating KNN with k=5...")
    acurracyScoreKnn5, f1Scoreknn5 = knnClassFunction(X_train, X_test, y_train, y_test, k=5)
    # print("------------------------------------------------")
    # Evaluar KNN con k=10
    # print("Evaluating KNN with k=10...")
    acurracyScoreKnn10, f1Scoreknn10 = knnClassFunction(X_train, X_test, y_train, y_test, k=10)
    # print("------------------------------------------------")
    # Evaluar KNN con k=5 y noormalizado
    # print("Evaluating KNN with k=5 & normalized...")
    acurracyScoreKnn5Norm, f1Scoreknn5Norm = knnClassFunction(X_train_scaled, X_test_scaled, y_train, y_test, k=5, normalized=True)
    # print("------------------------------------------------")
    # Evaluar KNN con k=10
    # print("Evaluating KNN with k=10 & normalized...")
    acurracyScoreKnn10Norm, f1Scoreknn10Norm = knnClassFunction(X_train_scaled, X_test_scaled, y_train, y_test, k=10, normalized=True)
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
    accuracyScoreLogRegres, f1ScoreLogRegres = logisticRegresionClassFunction(X_train, X_test, y_train, y_test)
    # print("------------------------------------------------")
    # print("Evaluating Logistic regresion with normalized data...")
    accuracyScoreLogRegresNorm, f1ScoreLogRegresNorm = logisticRegresionClassFunction(X_train_scaled, X_test_scaled, y_train, y_test, normalized=True)
    # print("##################################################")
    
    # print("################--Decision Tree Classifier--################")
    # print("Evaluating Decision Tree...")
    accuracyScoreDecTree, f1ScoreDecTree = decisionTreeClassFunction(X_train, X_test, y_train, y_test)
    # print("------------------------------------------------")
    # print("Evaluating Decision Tree with normalized data...")
    accuracyScoreDecTreeNorm, f1ScoreDecTreeNorm = decisionTreeClassFunction(X_train_scaled, X_test_scaled, y_train, y_test, normalized=True)
    # print("##################################################")
        
    # Definir los resultados como un diccionario
    results = {
        'KNN': {
            'K=5': {'Accuracy': acurracyScoreKnn5, 'F1 Score': f1Scoreknn5},
            'K=10': {'Accuracy': acurracyScoreKnn10, 'F1 Score': f1Scoreknn10},
            'Normalized data K=5': {'Accuracy': acurracyScoreKnn5Norm, 'F1 Score': f1Scoreknn5Norm},
            'Normalized data K=10': {'Accuracy': acurracyScoreKnn10Norm, 'F1 Score':f1Scoreknn10Norm}
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
    
    # Crear una lista de tuplas para cada fila de la tabla
    table_rows = []
    for classifier, params in results.items():
        for param, metrics in params.items():
            accuracy = metrics['Accuracy']
            f1_score = metrics['F1 Score']
            table_rows.append((classifier, param, accuracy, f1_score))
    
    # Imprimir la tabla utilizando tabulate
    print(tabulate(table_rows, headers=['CLASSIFIER', 'PARAMETERS', 'ACCURACY', 'F1Score'], tablefmt='grid'))
    
    
if __name__ == "__main__":
    main()
