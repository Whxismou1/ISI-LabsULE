#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 21:02:13 2024

@author: moasin
"""

import numpy as np


def main():
    np.random.seed(0)
    
    matrix = np.random.uniform(0, 3, (20, 20))
    
    print(matrix)
    
    print("++++++++++++++++++++++++++++++++++++++")
    coordenadasEntre1y2 = np.argwhere((matrix >= 1) & (matrix < 2))
    print(coordenadasEntre1y2)
    
    
    print("++++++++++++++++++++++++++++++++++++++")
    coordenadasFuerade1y2 = np.argwhere((matrix < 1) | (matrix > 2))
    print(coordenadasFuerade1y2)

    print("++++++++++++++++++++++++++++++++++++++")
    rounded_matrix = np.round(matrix)
    
    print(rounded_matrix)
    
    coordenadadDistintasDe1 = np.transpose(np.nonzero(rounded_matrix != 1))
    print(coordenadadDistintasDe1)
    
    

if __name__ == "__main__" :
    main()