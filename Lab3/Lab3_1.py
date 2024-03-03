#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 19:37:53 2024

@author: moasin

j. Calculate the sum of the elements of the first and third column of Matrix1 and
store them in a variable called mat_sum. Do it in only one line of code
"""

import numpy as np


def main():
    # Apartado A
    print("++++++++Apartado A++++++++")
    Matrix1 = np.array([[4, -2, 7], [9, 4, 1], [5, -1, 5]])
    print(Matrix1)

    # Apartado B
    print("++++++++Apartado B++++++++")
    Matrix2 = np.transpose(Matrix1)
    print(Matrix2)

    # Apartado C
    print("++++++++Apartado C++++++++")
    print(np.multiply(Matrix1, Matrix2))

    # Apartado D
    print("++++++++Apartado D++++++++")
    prodM1M2 = np.dot(Matrix1, Matrix2)
    print(prodM1M2)
    

    # Apartado E
    print("++++++++Apartado E++++++++")
    prodM2M1 = Matrix2 @ Matrix1
    print(prodM2M1)


    # Apartado F
    print("++++++++Apartado F++++++++")
    mat_corners = np.array(
        [Matrix1[0, 0], Matrix1[0, -1], Matrix1[-1, 0], Matrix1[-1, -1]])
    print(mat_corners)


    # Apartado G
    print("++++++++Apartado G++++++++")
    vec_max = np.array([sum(Matrix1[0]), sum(Matrix1[1]), sum(Matrix1[2])])
    print(Matrix1)
    global_max = np.max(Matrix1)
    print(vec_max, global_max)

    # Apartado H
    print("++++++++Apartado H++++++++")
    vec_min = np.min(Matrix1, axis=0)
    global_min = np.min(Matrix1)
    print(vec_min, global_min)
    
    print("++++++++Apartado I++++++++")
    vec_min = vec_min.reshape(-1, 1)
    vec_max = vec_max.reshape(1, -1)
    restul = vec_min @ vec_max
    print(restul)
    
    
    
    print("++++++++Apartado J++++++++")
    mat_sum = np.sum(Matrix1[:, [0, 2]])
    print(mat_sum)


if __name__ == "__main__":
    main()
