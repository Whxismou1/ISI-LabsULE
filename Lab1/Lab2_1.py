#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 17:19:26 2024

@author: moasin
"""



def calcTemp2Farenheint(tempC):
    tempF = (9/5) * tempC + 32
    return tempF

if __name__ == "__main__":
    temp = float(input("Introduce una temperatura en ªC: "))
    
    tempCoverted2F = calcTemp2Farenheint(temp)
    print("The temperature of {:.1f} degrees Celsius corresponds to {:.1f} degrees Farenheit".format(temp, tempCoverted2F))

