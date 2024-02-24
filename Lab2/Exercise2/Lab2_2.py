from Vector3D import Vector3D
import pickle

def main():
    vector = Vector3D()
    vector.printCoordinates()
    
    vector.setCoodinates(-6, 10, 5)
    vector.printCoordinates()
    
    another = Vector3D(5, -1, 0)
    
    vector.addVector(another)
    vector.printCoordinates()
    
    another.setCoodinates(-1, -1, -9)
    vector.subVector(another)
    
    vector.multiplyByScalar(3.5)
    vector.printCoordinates()
    

   
    modulus = vector.vectorModulus()
    print(f"Modulus: {modulus}")
    
    print("Vector que se guarda")
    vector.printCoordinates()
    rutaNRML = "./Lab2/Exercise2/vectorNor.txt"
    vector.storeVectorOnFile(rutaNRML)
    
    # rutaPKL = "./Lab2/Exercise2/vectorPKL.plk"
    # vector.storeVectorOnPickel(rutaPKL)
    
    # with open(rutaPKL, 'rb') as file:
    #     loaded = pickle.load(file)
    
    # print(f"Vector que se carga desde el archivo pickle {loaded.x} {loaded.y} {loaded.z}")
    
    
if __name__ == "__main__":
    main()