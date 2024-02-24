import random

ruta = "./Lab2/highscore.txt"

def updateHighScore(username, new_score):
    try:
        with open(ruta, "r+") as file:
            lines = file.readlines()
            file.seek(0)
            for line in lines:
                if username in line:
                    file.write(f"{username}:{new_score}\n")
                else:
                    file.write(line)
            file.truncate()
    except Exception as e:
        print(f"Error: {e}")



def getHighScoreUser(username):
    try:
        with open(ruta, "r") as file:
            lines = file.readlines()
            for line in lines:
                if username in line:
                    return line
    except:
        return 0


def exit():
    choice = input("Do you want to play again? (yes/no): ")
    if choice == "yes":
        main()
    else:
        print("Goodbye!")
        quit()


def startGuessGame(username, usernumber, secretNumber):
    numTries = usernumber
    highScore = getHighScoreUser(username)
    points = 0
    if highScore != 0:
        print(f"Highscore for {username} is: {highScore}")
    else:
        print("No highscore for this user")
    
    #num errors
    f = 0
    

    while numTries > 0:
        actualGuess = int(input(f"Try to guess whats the number im thinking of between 1 and {usernumber}: "))
        if actualGuess != secretNumber:
            numTries -= 1
            f += 1
            if actualGuess > secretNumber:
                print(f" Wrong! Try again! The number is lower than {actualGuess}. You have {numTries} tries left")
            else:
                print(f" Wrong! Try again! The number is higher than {actualGuess}. You have {numTries} tries left")
        else:
            print(f"Congratulations! You guessed the number {secretNumber}!")
            points = (usernumber/ pow(2, f))
            print(f"Your final score is: {points}")
            break

    if highScore != 0 and highScore is not None:
        if points > float(highScore.split(":")[1]):
            print(f"Congratulations! You have a new highscore! Your previous highscore was: {highScore}")
            updateHighScore(username, points)
        else:
            print(f"Your score was not enough to beat your highscore of {highScore}")
    else:
        with open(ruta, "a") as file:
            file.write(f"{username}:{points}\n")

        
    
    exit()
        

def getUserParams():
    userName = input("Hi! Whats your name?: ")
    userNumber = int(input("Please enter a number: "))
    secretNumber = random.randint(1, userNumber)
    
    return userName, userNumber, secretNumber

def main():
    username, usernumber, secretNumber = getUserParams()
    
    startGuessGame(username, usernumber, secretNumber)





if __name__ == "__main__":
    main()