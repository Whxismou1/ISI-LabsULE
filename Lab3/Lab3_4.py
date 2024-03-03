import timeit
import numpy as np


def getInputUser():
    numTimesL = int(
        input("Enter the number of times you want to run the program: "))
    return numTimesL


def listSquares(num):
    list = []
    for i in range(num):
        list.append(i**2)
    return list


def numpyArraySquares(num):
    squares = np.zeros(num, dtype=int)

    for i in range(num):
        squares[i] = i**2

    return squares


def vectorizingArray(num):
    vector = np.arange(0, num)
    squares = vector * vector
    return squares


def calculate_average_time(method, numTimes):
    setup = f"from __main__ import {method}"
    stmt = f"{method}({numTimes})"
    times = timeit.repeat(stmt, setup, repeat=5, number=1)
    avg_time = np.mean(times)
    return avg_time


def printMostFastestMethod(avgTimeList, avgTimeNumpyArray, avgTimeVectorized):
    print("---------------------------------------------------------------------------------")
    
    
    if avgTimeList < avgTimeNumpyArray and avgTimeList < avgTimeVectorized:
        print("The fastest method is method a (storing values in a list) with {}".format(
            avgTimeList))
    elif avgTimeNumpyArray < avgTimeList and avgTimeNumpyArray < avgTimeVectorized:
        print("The fastest method is method b (storing values in a NumPy array) with {}".format(
            avgTimeNumpyArray))
    else:
        print("The fastest method is method c (vectorized operation) with {}".format(
            avgTimeVectorized))


def main():
    numTimes = getInputUser()

    avgTimeList = calculate_average_time("listSquares", numTimes)
    print(
        f"Average time for method a (storing values in a list): {avgTimeList} seconds")

    # Calculate average time for method b
    avgTimeNumpyArray = calculate_average_time("numpyArraySquares", numTimes)
    print(
        f"Average time for method b (storing values in a NumPy array): {avgTimeNumpyArray} seconds")

    # Calculate average time for method c
    avgTimeVectorized = calculate_average_time("vectorizingArray", numTimes)
    print(
        f"Average time for method c (vectorized operation): {avgTimeVectorized} seconds")

    printMostFastestMethod(avgTimeList, avgTimeNumpyArray, avgTimeVectorized)


if __name__ == "__main__":
    main()
