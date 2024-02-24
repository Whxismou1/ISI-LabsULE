#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 17:46:08 2024

@author: moasin
"""


if __name__ == "__main__":
    listOfNumbers = []
    powOfNumbers = []
    while True:
        num = input("Inserte un numero: ")
    
        try:
            num = int(num)
        except ValueError:
            print("ERROR: Debes introducir un número.")
        else:
            listOfNumbers.append(num)
            powOfNumbers.append(num ** 2)
        
        if(num < 0):
            break


    print("Lista final: ", listOfNumbers)
    print("Lista de cuadrados: ", powOfNumbers)
    print("Suma total de los cuadrados: ", sum(powOfNumbers))
    
    