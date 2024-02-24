import math
import pickle

class Vector3D:
    
    def __init__(self, x=0, y=0, z=0):
        self.x = x
        self.y = y
        self.z = z

    def setCoodinates(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z
        
    def printCoordinates(self):
        print(f"Coordinates: ({self.x}, {self.y}, {self.z})")
        
    def addVector(self, other):
        self.x += other.x
        self.y += other.y
        self.z += other.z
        
    def subVector(self, other):
        self.x -= other.x
        self.y -= other.y
        self.z -= other.z
    
    def multiplyByScalar(self, scalar):
        self.x *= scalar
        self.y *= scalar
        self.z *= scalar
        
    def vectorModulus(self):
        return math.sqrt((self.x**2) + (self.y**2) + (self.z**2))
        
        
    def storeVectorOnFile(self, filename):
        with open(filename, 'w') as file:
            file.write(f"{self.x}, {self.y}, {self.z}")
    
    
    def storemodulusOnFile(self, filename):
        with open(filename, 'w') as file:
            file.write(f"{self.x}, {self.y}, {self.z} : {self.vectorModulus()}")
    
    
    def storeVectorOnPickel(self, filename):
        with open(filename, 'wb') as file:
            pickle.dump(self, file)
            
    def storemodulusOnPickel(self, filename):
        with open(filename, 'wb') as file:
            pickle.dump(self.vectorModulus(), file)