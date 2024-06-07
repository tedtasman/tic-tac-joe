"""
06/05/2024
@authors: (Managing Director) Benjamin Rodgers and (CEO) Theodore Tasman

This is the main AI training file. It builds the model, and iteratively updates it depending on the game results.

"""

import os
import board as bd
import ioBoard as io
import numpy as np
import tensorflow as tf
import random as rd
import time
import curses

version = '1.2.0'

def __buildModel():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(9,)),  # Input layer (9 cells in Tic Tac Toe)
        #tf.keras.layers.Dropout(0.5),  # Dropout layer to prevent overfitting
        tf.keras.layers.Dense(64, activation='relu'),  # Hidden layer
        #tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(64, activation='relu'),  # Hidden layer 2 
        #tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(9, activation='softmax')  # Output layer (9 possible actions)
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy')

    return model


# Board -> Input function to convert the board state to a one off value
def boardStateValue(board):
    
    vectorInput = np.zeros(9) #returns array of given shape (9) and its filled with zeroes

    # iterate through rows & cols of board
    for row in range(3):

        for col in range(3):
            # calculate index of 1x9 vector from 3x3 grid
            # if square is an X:
            if board.grid[row][col] == 1:
                # set cooresponding index to 1
                vectorInput[3*row + col] = 1
            # if square is an O
            elif board.grid[row][col] == 2:
                # set cooresponding index to -1
                vectorInput[3*row + col] = -1

    return vectorInput


# Choose action (board, model) -> choose an action based on prediction
def getAction(board, model):
    
    vectorInput = boardStateValue(board)
    # get probabilities of each action
    probabilities = model.predict(vectorInput.reshape(1,-1), verbose=0)
    # sort actions by best probability
    sortedProbabilites = np.argsort(probabilities[0])[::-1]

    # iterate through actions
    for action in sortedProbabilites:
 
        row, col = divmod(action, 3)

        # if current action is valid
        if board.validMove(row, col):
            return row, col # return current action
        
    raise ValueError("No valid moves but game is not over.")


# determine winner
def winnerDeter(board):
    win = board.gameWon()
    if win == 1:
        return -1
    elif win == 2:
        return 1
    else:
        return 0
    

def trainModel(version=None, iteration=None):
    """

    This is the training function we have described for the model. It allows the model to gain
    experience on the game and learn. This is reinforcement learning at its finest. You can set iterations
    to make it more and more experienced. 
    
    """

    # check if new model needs to be made
    offset = 0
    if version and iteration:
        offset = int(iteration)
        model = loadModel(version, iteration)

    else:
        model = __buildModel()

    iterations = int(input("Set iterations: "))
    saveInterval = int(input("Set save interval: "))

    # run training
    for i in range(iterations):  # Play 1,000 games
        board = bd.Board()

        # for at most 9 moves
        for j in range(9):

            # get vector of board state
            vectorInput = boardStateValue(board) 
            # get move
            row, col = getAction(board, model)
            # play move
            board.playMove(row, col)
            # check for winner
            result = winnerDeter(board)

            # Update the model
            # get probibilities for each move
            probabilities = model.predict(vectorInput.reshape(1,-1), verbose=0)[0]
            # set target to current probabilities
            target = probabilities[:]
            # update probability of chosen square to reflect result
            target[3 * row + col] = result
            # retrain model
            model.fit(vectorInput.reshape(1,-1), np.array([target]), verbose=0)

            # escape to next round if there was a winner
            if result != 0:
                break
                
        print("Iteration {} ended in {} moves with result {}".format(i + offset + 1, j + 1, result))
        
        if (i + offset + 1) % saveInterval == 0:
            saveModel(model, 'joe-v{}-i{}'.format(version, (i + offset + 1)))

    return model


def __runUserPlay(stdscr, model):

    """
    This is a function that allows the user to play against the model that has been trained.
    Very straight forward.
    """
    
    play = 'foo'
    
    while True:  # Play games until the user decides to quit

        board = bd.Board()
        joeTurn = rd.randint(1,2)
        curses.curs_set(0) # hide the cursor
        board.drawBoard(stdscr)

        while board.gameWon() == 0:
            if board.nextMove == joeTurn:  # AI's turn
                time.sleep(0.2)
                stdscr.addstr(7, 0, "Joe's Move...")
                stdscr.refresh()
                time.sleep(1)
                stdscr.refresh()
                row, col = getAction(board, model)
                board.playMove(row, col)
                board.drawBoard(stdscr)
            else:  # Human's turn
                io.getMove(stdscr, board, True)

        # Print the result of the game
        if board.gameWon() == joeTurn:
            stdscr.addstr(7, 0, "Joe wins!")
        elif board.gameWon() == 0:
            stdscr.addstr(7, 0, "It's a draw!")
        else:
            stdscr.addstr(7, 0, "You win!")

        # Ask the user if they want to play again
        stdscr.addstr("\nDo you want to play again? (yes/no) ")
        # repeat until valid answer
        while True:
            play = stdscr.getkey()
            if play.lower() in ["yes", "y", "no", "n"]:
                break
            stdscr.addstr("\nInvalid input. Please enter 'yes' or 'no'.\n")
            stdscr.refresh()            

        if play.lower() in ['no', 'n']:
            break


def playUser(model):
    curses.wrapper(__runUserPlay, model)


def saveModel(model, name):

    """
    This saves a model to a desired path for later use.
    """

    model.save("./models/{}.keras".format(name))
    print("Model ./models/{}.keras saved.".format(name))


def loadModel(version, iteration):
    """
    This loads a model from the desired path for use. 
    """
    #name = getFileName(version, iteration)
    if type(version) is not str:
        raise ValueError("Version must be input as a string")
    
    if type(iteration) is not int:
        raise ValueError("Iteration must be input as an int")

    modelPath = "./models/joe-v{}-i{}.keras".format(version, str(iteration))
    if os.path.isfile(modelPath):
        model = tf.keras.models.load_model(modelPath)
        print("Model {} loaded with optimizer state.".format(modelPath))
        return model
    else:
        print("Sorry pal, that file doesn't exist.")
        return None



def deleteAllModels___RED_BUTTON():

    confirm = input('Are you sure you want to delete all models? This cannot be undone.\nType "CONFIRM" to proceed: ')
    if confirm not in ['CONFIRM']:
        return

    for model in os.listdir('./models/'):
        filePath = os.path.join('./models/', model)
        try:
            if filePath != './models/joeRandom.keras/':
                os.remove(filePath)
                print("Deleted model: {}".format(filePath))
            
        except Exception as e:
            print("Failed to delete {}. Reason: {}".format(filePath, e))


def playModels(model1, model2):

    games = 0
    # repeat until valid answer
    while not isinstance(games, int) or int(games) < 1:
        games = int(input("\nHow many games to play? "))
            
    wins = 0

    for i in range(games): 
        model1Turn = rd.randint(1,2)
        board = bd.Board()
        while board.gameWon() == 0:
            if board.nextMove == model1Turn:  # Model 1's turn
                row, col = getAction(board, model1)
                board.playMove(row, col)
            else:  # Model 2's turn
                row, col = getAction(board, model2)
                board.playMove(row, col)

        # Update the win count
        if board.gameWon() == model1Turn:
            wins += 1

        print("MODEL 1 STATS: \t Wins: {} \t Games Played: {} \t Win/Loss: {:0.3f}".format(wins, (i + 1), wins / (i + 1)))


'''def getFileName(version, iteration):

    if type(version) is not str:
        raise ValueError("Version must be input as a string")
    
    if type(iteration) is not int:
        raise ValueError("Iteration must be input as an int")

    testname = "./models/joe-v{}-i{}.keras".format(version, str(iteration))
    if os.path.isfile(testname):
        return testname
    else:
        print("Sorry pal, that file doesn't exist.")
        return None'''
    







# ============= MAIN ====================================================================================
if __name__ == "__main__":
    i1000 = loadModel('1.0.1', 1000)
    i250 = loadModel('1.0.1', 250)
    playUser(i250)