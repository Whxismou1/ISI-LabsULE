import math

def printMenu():
    print("\n------------Operations------------")
    print("1. Add an arbitrary number of values")
    print("2. Subtract two values")
    print("3. Multiply an arbitrary number of values")
    print("4. Divide two values")
    print("5. Calculate the value of one number raised to another")
    print("6. Calculate the natural logarithm of a number")
    print("7. Exit")

def getOption(option):
    if option in [1, 3]:
        values = list(map(float, input("Enter the values separated by a space: ").split()))
        if option == 1:
            print(f"Result of ∑({values}) = {addArbitraryNumbers(values)}")
        elif option == 3:
            print(f"Result of ∏({values}) = {multiplyArbitraryNumbers(values)}")
    elif option in [2, 4, 5]:
        num1 = float(input("Enter the first number: "))
        num2 = float(input("Enter the second number: "))
        
        if option == 2:
            print(f"Result of {num1} - {num2} = {subtraction(num1, num2)}")
            
        elif option == 4:
            print(f"Result of {num1} / {num2} = {divide2Nums(num1, num2)}")
        elif option == 5:
            print(f"Result of {num1} - {num2} = {power(num1, num2)}")
    else:
        num = float(input("Enter the number: "))
        if option == 6:
            print(f"Result of ln({num}) = {logNatural(num)}")
        
        
        
def addArbitraryNumbers(numbers):
    return sum(numbers)

def subtraction(a, b):
    return a - b

def multiplyArbitraryNumbers(numbers):
    result = 1
    for number in numbers:
        result *= number
    return result

def divide2Nums(a, b):
    if b == 0:
        return "Invalid input"
    return a / b

def power(base, exp):
    return base ** exp

def logNatural(num):
    if(num <= 0):
        return "Invalid input"
    return math.log(num)


def main():
    while True:
        printMenu()
        option = int(input("Option: "))
        if option == 7:
            print("Exiting...")
            
            break
        getOption(option)
    
    
    
    

if __name__ == "__main__":
    main()