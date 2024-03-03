import numpy as np


def getRandomMatrix():
    return np.random.randint(-10, 10, size=(10, 4))


def printMatrices(matrixA, eucledianMatrix):
    print("Matrix A: \n", matrixA)
    print("\nEucledian Matrix: \n", eucledianMatrix)


def getEucledianDistance(matrixA, eucledianMatrix):
    for i in range(len(matrixA)):
        for j in range(len(matrixA)):
            diff = matrixA[i] - matrixA[j]
            dist = np.sqrt(np.sum(diff**2))
            eucledianMatrix[i][j] = dist

            if dist < 10:
                print(
                    "The Euclidean distance between vectors {} and {} is {}".format(i, j, dist))


def main():
    np.random.seed(0)

    matrixA = getRandomMatrix()

    eucledianMatrix = np.zeros((10, 10))

    getEucledianDistance(matrixA, eucledianMatrix)

    printMatrices(matrixA, eucledianMatrix)


if __name__ == "__main__":
    main()
