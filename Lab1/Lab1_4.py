#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 18:10:49 2024

@author: moasin
"""


def getFirstChar2UpperCase(sentence):
    listOfWords = sentence.split()
    
    for i in range(len(listOfWords)):
        listOfWords[i] = listOfWords[i].capitalize()
    
    return " ".join(listOfWords)


def getCharOnEvenPos2UpperCase(sentence):
    listResult = []
    for i in range(len(sentence)):
        if (i % 2) == 0:
            listResult.append(sentence[i].upper())
        else:
            listResult.append(sentence[i])
    
    return "".join(listResult)

if __name__ == "__main__":
    
    sentence = input("Introduce una frase: ")
    
    print("The sentence into uppercase: " + sentence.upper())
    print("The sentence into lowercase: " + sentence.lower())
    print("The first character of each word into uppercase: " +  getFirstChar2UpperCase(sentence))
    print("the characters that are in even positions into uppercase: " + getCharOnEvenPos2UpperCase(sentence))
   