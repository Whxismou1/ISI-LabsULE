#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 18:02:38 2024

@author: moasin
"""


def calcFactorialRec(num):
    if (num == 1):
        return 1
    else:
        return num * calcFactorialRec(num-1)


def calcFactNorm(num):
    result = 1
    while(num > 1):
        result *= num
        num -= 1
    return result


if __name__ == "__main__":
    num = int(
        input("Introduce el numero del que quieras hallar el factorial [n!]: "))

    factRec = calcFactorialRec(num)
    factNorm = calcFactNorm(num)

    print("Recursivo: {}!= {}".format(num, factRec))
    print("Normal: {}!= {}".format(num, factNorm))
