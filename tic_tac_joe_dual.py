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
import dualModel as dm

build = '5.0.0'

# ======== HYPERPARAMETERS ===========

alpha = 0.01     # learning rate
gamma = 0.9     # discount factor
epsilon = 1.0   # exploration rate

epsilonDecay = 0.995
epsilonMin = 0.1

tieReward = 0.2

#============================

def __buildModel():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(10,)),  # Input layer (9 cells in Tic Tac Toe)
        tf.keras.layers.Dense(64, activation='relu'),  # Hidden layer
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(64, activation='relu'),  # Hidden layer 2 
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(64, activation='relu'),  # Hidden layer 3
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(9, activation='softmax')  # Output layer (9 possible actions)
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(alpha), loss='categorical_crossentropy')

    return model


# Board -> Input function to convert the board state to a one off value
def boardStateValue(board):
    
    vectorInput = np.zeros(10) #returns array of given shape (9) and its filled with zeroes

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
    
    # set nextmove to correct value
    vectorInput[9] = 1 if board.nextMove == 1 else -1

    return vectorInput


# Choose action (board, model) -> choose an action based on prediction
'''def getAction(vectorInput, board, model):
    
    # get probabilities of each action
    probabilities = model.predict(vectorInput.reshape(1,-1), verbose=0)
    # sort actions by best probability
    sortedIndices = np.argsort(probabilities[0])[::-1]

    # iterate through actions
    for action in sortedIndices:
 
        row, col = divmod(action, 3)

        # if current action is valid
        if action < 9 and board.validMove(row, col):
            return row, col # return current action
        
    raise ValueError("No valid moves but game is not over.")'''


# determine winner
def winnerDeter(board):
    win = board.gameWon()
    if win == 1:
        return -1
    elif win == 2:
        return 1
    else:
        return 0
    

# train model (str version, int iteration) -> model
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
        dualModel = dm.DualModel()

    iterations = int(input("Set iterations: "))
    saveInterval = int(input("Set save interval: "))
    print('\n====================================================\n')
    print('Running {} iterations with save interval {}... (ctrl + C to exit)\n'.format(iterations, saveInterval))
    # run training
    for i in range(iterations - offset):  # Play 1,000 games
        
        board = bd.Board()
        # get vectorInput before game (all 0s)
        vectorInput = boardStateValue(board) 
        boardStates = [(vectorInput, None)]

        # for at most 9 moves
        for j in range(9):



            # get qValues (probablities)
            currentModel = dualModel.getCurrentModel(board)
            qValues = currentModel.predict(vectorInput.reshape(1,-1), verbose=0)[0]

            # EXPLORE (pick random square)
            if rd.random() <= max(epsilonMin, epsilon * epsilonDecay * (i + 1)):

                # loop until valid move generated (capped at 12 to avoid infinte loop edge case)
                for _ in range(12): 
                    
                    # generate the random move
                    row, col = np.random.randint(0,3), np.random.randint(0,3)

                    # test validity, escape if valid
                    if board.validMove(row, col):
                        action = 3 * row + col
                        break
            
            # EXPLOIT (use model probabilities)
            else:
                # get best action
                action = currentModel.bestValidAction(board, qValues)
                # convert to row, col
                row, col = divmod(action, 3)

            # play move
            board.playMove(row, col)

            # save move for backprop
            boardStates.append((vectorInput, action))

            # check for winner
            result = winnerDeter(board)

            # set target to current probabilities
            target = qValues[:]

            # if there was a winner
            if result != 0:
                
                # reward based on result
                target[3 * row + col] = result

            # if current game was a tie
            elif board.gameWon() == 3:
                # give tieReward (hyperparameter)
                target[3 * row + col] = tieReward

            # retrain model
            currentModel.fit(vectorInput.reshape(1,-1), np.array([target]), verbose=0)
            
            # update board state
            vectorInput = boardStateValue(board) 

            # escape to next round if there was a winner
            if result != 0:
                break
        
        finalModel = currentModel

        if finalModel == dualModel.xModel:
            otherModel = dualModel.oModel
        else:
            otherModel = dualModel.xModel

        turn = 1
        for state, action in reversed(boardStates):
            

            # if same model as final model 
            if turn == 1:
                currentModel = finalModel
            else:
                currentModel = otherModel
    
            # get qvalues for that state
            qValues = currentModel.predict(state.reshape(1,-1), verbose=0)[0]
            target = qValues[:]

            # update targer of action to final result * gamma
            target[action] = result * gamma
            
            # retrain currentmodel
            currentModel.fit(vectorInput.reshape(1,-1), np.array([target]), verbose=0)
        
            result *= -gamma
            turn *= -1

        print("\rIteration {} ended in {} moves. {}!".format(i + offset + 1, j + 1, 'X wins' if result == 1 else 'O wins' if result == -1 else "A draw"), end='')
        
        if (i + offset + 1) % saveInterval == 0:
            saveModel(model, 'joe-v{}-i{}'.format(build, (i + offset + 1)))

    return model

#we need to figure out how to save the dual model

# get best action (board, qValues) -> qvalue
def bestValidAction(board, qValues):

    # sort qValues
    sortedIndices = np.argsort(qValues)[::-1]

    for i in sortedIndices:
        
        # get row/column from qValue
        row, col = divmod(i, 3)

        # if current action is valid
        if board.validMove(row, col):

            return i # return current qValue index
        
    return -1


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
                
                # graphics
                time.sleep(0.2)
                stdscr.addstr(7, 0, "Joe's Move...")
                stdscr.refresh()
                time.sleep(1)
                stdscr.refresh()

                # get current board State
                vectorInput = boardStateValue(board)

                # get qValues from state
                qValues = model.predict(vectorInput.reshape(1,-1), verbose=0)[0]

                # get best action
                action = bestValidAction(board, qValues)

                # play move
                row, col = divmod(action, 3)
                board.playMove(row, col)

                # update screen
                board.drawBoard(stdscr)

            else:  # Human's turn
                io.getMove(stdscr, board, True)

        # Print the result of the game
        if board.gameWon() == joeTurn:
            stdscr.addstr(7, 0, "Joe wins!")
        elif board.gameWon() == 3:
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
    # figure out a way to save both models together. maybe just by name. maybe put it in the dualModel module
    model.save("./models/{}.keras".format(name))
    print("\nMODEL SAVED:\t./models/{}.keras\n".format(name))
    

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

        # randomly decide first move
        model1Turn = rd.randint(1,2)
        # make new board
        board = bd.Board()

        # until a winner is decided
        while board.gameWon() == 0:

            # get current state
            vectorInput = boardStateValue(board)

            # decide which model is next
            if board.nextMove == model1Turn:  # Model 1's turn

                currentModel = model1

            else:  # Model 2's turn
                
                currentModel = model2
            
            # get qValues from state
            qValues = currentModel.predict(vectorInput.reshape(1,-1), verbose=0)[0]

            # get best action
            action = bestValidAction(board, qValues)

            # play move
            row, col = divmod(action, 3)
            board.playMove(row, col)

        # Update the win count
        if board.gameWon() == model1Turn:
            wins += 1

        print("MODEL 1 STATS: \t Wins: {} \t Games Played: {} \t Win/Loss: {:0.3f}".format(wins, (i + 1), wins / (i + 1)))


# ============= MAIN ====================================================================================
if __name__ == "__main__":
    trainModel()
    '''joeRandom = loadModel('0', 0)
    a1000 = loadModel('1.0.0', 1000)
    b500 = loadModel('1.2.0', 500)
    b1000 = loadModel('1.2.0', 1000)
    c500 = loadModel('2.0.0', 500)
    d500 = loadModel('2.1.0', 500)
    e500 = loadModel('2.2.0', 500)'''
