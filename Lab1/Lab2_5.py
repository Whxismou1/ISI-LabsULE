#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 17:38:07 2024

@author: moasin
"""


def getLetterOfDNI(numDNI):
    
    lista = ['T', 'R', 'W', 'A', 'G', 'M', 'Y', 'F', 'P', 'D', 'X', 'B', 'N', 'J', 'Z', 'S', 'Q', 'V', 'H', 'L', 'C', 'K', 'E']
    
    remaining = numDNI % 23
    
    print(remaining)
    
    if numDNI % 2==0:
        return lista[remaining]
    else:
        return """"""
    
    



if __name__ == "__main__":
    
    numDNI = int(input("Introduce el DNI sin la letra: "))
    
    letter = getLetterOfDNI(numDNI)
    
    print("La letra correspondiente a tu DNI {} es: {}".format(numDNI,letter))