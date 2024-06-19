"""
06/11/2024
@authors: (Managing Director) Benjamin Rodgers and (CEO) Theodore Tasman

This is the main AI training file. It builds the model, and iteratively updates it depending on the game results.

"""

import curses
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # only show tensorflow errors
import time
import numpy as np
import board as bd
import random as rd
import ioBoard as io
import tensorflow as tf
import matplotlib.pyplot as plt

BUILD = '6.3.1'



# ======== HYPERPARAMETERS ===========

ALPHA = 0.01     # learning rate
GAMMA = 0.9     # discount factor

EPSILON = 1.0   # exploration rate
EPSILON_DECAY = 0.995 # decay rate for epsilon
EPSILON_MIN = 0.1 # minimum epsilon

TIE_REWARD = 0.5  # reward for tie
BLUNDER_PENALTY = -1 # penalty for blunder (missed win or unforced loss)
INVALID_PENALTY = 0 # penalty for invalid move
SCORE_MULTIPLIER = 1 # multiplier for score



# ============= FUNCTIONS ==============================================================================

'''
builds the model with the given alpha

returns model object
'''
def __buildModel(alpha):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(9,)),  # Input layer (9 cells in Tic Tac Toe)
        tf.keras.layers.Dense(64, activation='relu'),  # Hidden layer
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(64, activation='relu'),  # Hidden layer 2 
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(64, activation='relu'),  # Hidden layer 3
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(9, activation='tanh')  # Output layer (9 possible actions)
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(alpha), loss='mse')

    return model



'''
saves the model with the given build, alpha, gamma, and iteration

returns nothing
'''
def saveModel(model, build, alpha, gamma, iteration):

    # check if build directory exists
    if not os.path.exists(f"./{build}/"):
        os.makedirs(f"./{build}/")
    
    # save model
    model.save(f"./{build}/a{alpha}_g{gamma}_i{iteration}.keras")


    print(f"MODEL SAVED: a{alpha}_g{gamma}_i{iteration}.keras in {build} directory.")



'''
Loads the model with the given build, alpha, gamma, and iteration

returns model object
'''
def loadModel(build=None, alpha=None, gamma=None, iteration=None):

    # if not given, get from user
    if not (build and alpha and gamma and iteration):
        build = input("Enter build: ")
        alpha = float(input("Enter alpha (0.1 - 0.01): "))
        gamma = float(input("Enter gamma (0.9 - 0.99): "))
        iteration = int(input("Enter iteration: "))

    print(f"MODEL LOADED: a{alpha}_g{gamma}_i{iteration}.keras in {build} directory.")

    return tf.keras.models.load_model(f"./{build}/a{alpha}_g{gamma}_i{iteration}.keras")



'''
loads the model with the current hyperparameters and build, given iteration

returns dualModel object
'''
def loadCurrent(iteration):
    return loadModel(BUILD, ALPHA, GAMMA, iteration)



'''
generates all possible future states given a vector representation of the current state and the next move

returns list of future states
'''
def predictFutureStates(vectorInput, nextMove):

    futureStates = []

    # for every space on the board
    for i in range(9):

        # if other player can play there
        if vectorInput[i] == 0:

            # play that move on a copy of the board
            futureVector = vectorInput.copy()
            futureVector[i] = nextMove

            # add the future state to the list
            futureStates.append(futureVector)

    return futureStates



'''
checks for a win given a vector representation of the board

returns 1 for X win, -1 for O win, 2 for tie, 0 for no win

'''
def vectorWin(vector):
    
        # check rows
        for i in range(3):
    
            if sum(vector[3*i:3*i+3]) == 3:
                return 1
    
            elif sum(vector[3*i:3*i+3]) == -3:
                return -1
    
        # check columns
        for i in range(3):
    
            if sum(vector[i::3]) == 3:
                return 1
    
            elif sum(vector[i::3]) == -3:
                return -1
    
        # check diagonals
        if sum(vector[0:9:4]) == 3 or sum(vector[2:7:2]) == 3:
            return 1
    
        elif sum(vector[0:9:4]) == -3 or sum(vector[2:7:2]) == -3:
            return -1

        # if all spaces filled (tie)
        if 0 not in vector:
            return 2

        return 0



'''
Gets best moves for exploitation, sets blunder penalty for missed wins and unforced losses

returns tuple: (target qValues, action)
'''
def exploit(vectorInput, currentModel, board, blunderPenalty=0, tieReward=0):

    # get qValues based on current turn
    qValues = currentModel.predict(vectorInput.reshape(1,-1), verbose=0)[0]

    # get valid indices
    validActions =  [i for i in range(9) if board.validMove(*divmod(i, 3))]

    # check for immediate win/loss
    for action in validActions:
        
        # play move on copy of board
        futureVector = vectorInput.copy()
        futureVector[action] = board.nextMove

        # if current agent wins with this move
        if vectorWin(futureVector) == board.nextMove:

            qValues[action] = 1

        # if current agent ties with this move
        elif vectorWin(futureVector) == 2:

            qValues[action] = tieReward
    
    # if no immediate win
    if 1 not in qValues:
        
        # iterate through best valid actions checking for a potential loss
        for action in validActions:

            # play move on copy of board
            futureVector = vectorInput.copy()
            futureVector[action] = board.nextMove

            # get opponent's future states
            doubleFutureStates = predictFutureStates(futureVector, -board.nextMove)
            for future in doubleFutureStates:

                # if opponent can win
                if vectorWin(future) == -board.nextMove:

                    qValues[action] = blunderPenalty
                    break
    
    # set all invalid moves to negative infinity
    '''for i in range(9):
        if i not in validActions:
            qValues[i] = INVALID_PENALTY'''

    # get best valid action
    for action in np.argsort(qValues)[::-1]:
        if board.validMove(*divmod(action, 3)):
            break

    # play move
    board.playMove(*divmod(action, 3))

    # return adjusted qValues
    return qValues, action



'''
Checks for a win or tie available. If none, plays a random move.
If win, sets qValue of action to 1. If tie, sets qValue of action to tie reward.

If random is set to True, returns tuple with smart, qValues, and action. Otherwise,

'''
def explore(vectorInput, currentModel, board, tieReward, random=False):
    
    # get qValues based on current turn
    qValues = currentModel.predict(vectorInput.reshape(1,-1), verbose=0)[0]

    # get valid indices
    validActions =  [i for i in range(9) if board.validMove(*divmod(i, 3))]

    # set smart to False to indicate random move
    smart = False

    # initailize smart action to None
    smartAction = None

    # loop through best actions to check for win or tie
    for action in validActions:
        
        # play move on copy of board
        futureVector = vectorInput.copy()
        futureVector[action] = board.nextMove

        # if current agent wins with this move
        if vectorWin(futureVector) == board.nextMove:

            qValues[action] = 1
            smart = True # set smart to True to indicate a smart move can be made
            smartAction = action # set smart action to current action

        # if current agent ties with this move
        elif vectorWin(futureVector) == 2:

            qValues[action] = tieReward

            # if no winning move, set smart action to current action
            if not smartAction:
                smart = True # set smart to True to indicate a smart move can be made
                smartAction = action # set smart action to current action

    '''# set all invalid moves to negative infinity
    for i in range(9):
        if i not in validActions:
            qValues[i] = INVALID_PENALTY'''

    # if there is a smart move
    if smart:
        action = smartAction
    
    # if no smart move
    else:

        # get random valid action
        action = rd.choice(validActions)

    # play move
    board.playMove(*divmod(action, 3))
    
    # return tuple with smart if random is set to True
    if random:
        return smart, qValues, action
    
    # return adjusted qValues
    else:
        return qValues, action


'''
propogates the result back through the model

returns nothing
'''
def backpropagate(boardStates, model, result, gamma):

    for state, action in reversed(boardStates):

        # get qValues for that state
        qValues = model.predict(state.reshape(1,-1), verbose=0)[0]
        target = qValues.copy()

        # update target of action to final result * gamma
        target[action] = result * gamma

        # refit model
        model.fit(state.reshape(1,-1), target.reshape(1,-1), verbose=0)

        # decrease result by gamma
        result *= gamma


'''
train the model using the hyperparameters defined above or custom hyperparameters if override is set to True.

input (optional): override (bool) - if True, use custom hyperparameters - default is False
input (optional): backprop (bool) - if True, backpropagate results - default is True
returns dualModel object
'''
def trainModel(load=False, override=False, backprop=True):

    # if load, get model details to load
    if load:
        modelBuild = input("Enter build: ")
        modelAlpha = float(input("Enter alpha (0.1 - 0.01): "))
        modelGamma = float(input("Enter gamma (0.9 - 0.99): "))
        modelIteration = int(input("Enter iteration: "))
        model = loadModel(modelBuild, modelAlpha, modelGamma, modelIteration)
        offset = modelIteration

    # if override, get custom hyperparameters
    if override:
        alpha = float(input("Set alpha: "))
        gamma = float(input("Set gamma: "))
        epsilon = float(input("Set epsilon: "))
        epsilonDecay = float(input("Set epsilon decay: "))
        epsilonMin = float(input("Set epsilon min: "))
        tieReward = float(input("Set tie reward: "))
        blunderPenalty = float(input("Set blunder penalty: "))

    # else, use globals
    else:
        alpha = ALPHA
        gamma = GAMMA
        epsilon = EPSILON
        epsilonDecay = EPSILON_DECAY
        epsilonMin = EPSILON_MIN
        tieReward = TIE_REWARD
        blunderPenalty = BLUNDER_PENALTY

    # build model
    if not load:
        model = __buildModel(alpha)
        offset = 0

    # define training
    iterations = int(input("Set iterations: "))
    saveInterval = int(input("Set save interval: "))
    print('\n====================================================\n')
    print('Running {} iterations with save interval {}... (ctrl + C to exit)\n'.format(iterations, saveInterval))

    wins = 0
    losses = 0
    ties = 0
    winPercentage = 0
    winPercentages = []
    averageMoves = 0

    # run training
    for i in range(iterations - offset):
            
        # create new board
        board = bd.Board()

        # get vector representation of empty board
        vectorInput = np.zeros(9)
        
        if backprop:
            # list to store states and actions for backpropagation
            boardStates = [(vectorInput, None)]

        # model goes first on even iterations and second on odd iterations
        if i % 2 == 0:
            modelTurn = 1
        else:
            modelTurn = -1

        # repeat for at most 9 moves
        for j in range(9):

            # if model's turn
            if board.nextMove == modelTurn:

                smart = True

                # EXPLORE (pick random square)
                if rd.random() <= max(epsilonMin, epsilon * (epsilonDecay ** i)):

                    # call explore function, plays move and returns targetQValues
                    targetQValues, action = explore(vectorInput, model, board, tieReward)
                
                # EXPLOIT (use model probabilities)
                else:

                    # call exploit function, returns targetQValues
                    targetQValues, action = exploit(vectorInput, model, board, blunderPenalty, tieReward)    

                    if backprop:
                        # record state and action for backpropagation
                        boardStates.append((vectorInput, action))            
            
            # if not model's turn
            else:

                # call explore function for a random move, plays move and returns targetQValues
                smart, targetQValues, action = explore(vectorInput, model, board, tieReward, True)

            # only refit for smart moves, not random
            if smart:
                # refit the model
                model.fit(vectorInput.reshape(1,-1), targetQValues.reshape(1,-1), verbose=0)

            # check if game is over
            winner = board.gameWon()

            # if model wins
            if winner == modelTurn:
                wins += 1
                break
            
            # if model loses
            elif winner == -modelTurn:
                losses += 1
                break
            
            # if tie
            elif winner == 2:
                ties += 1
                break
            # if game continues; get new vector representation of board
            else:

                vectorInput = board.vector

        # backpropagate result
        if backprop:
            # if model wins -> reward
            if winner == modelTurn:
                backpropagate(boardStates, model, 1, gamma)

            # if model loses -> punish
            elif winner == -modelTurn:
                backpropagate(boardStates, model, -1, gamma)

            # if tie -> tie reward
            else:
                backpropagate(boardStates, model, tieReward, gamma)

        # record progress
        averageMoves = (averageMoves * i + j) / (i + 1)
        winPercentage = (wins + 0.5 * ties) / (i + 1)
        winPercentages.append(winPercentage)
        print(f"\033[1K\r\033[0KAfter {i + offset + 1} iterations: \t W-L-T: {wins}-{losses}-{ties} \t Win/Loss: {(wins + 0.5 * ties) / (i + 1):.3f} \t Average Moves: {averageMoves:.1f}", end='')

        if (i + 1) % saveInterval == 0:
            if not backprop:
                print('\n')
                saveModel(model, BUILD + 'nb', alpha, gamma, i + offset + 1)
            print('\n')
            saveModel(model, BUILD, alpha, gamma, i + offset + 1)
            print('\n')

            # plot win percentage over time
            plt.plot(winPercentages)
            plt.title('Win Percentage Over Time')
            plt.xlabel('Iteration')
            plt.ylabel('Win Percentage')
            plt.show()

    print('\n\nTraining complete!\n')

    return model


'''runs AI vs user game in curses wrapper'''
def __runUserPlay(stdscr, model):
    
    play = 'foo'
    
    while True:  # Play games until the user decides to quit

        board = bd.Board()
        joeTurn = rd.choice([-1, 1])
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

                # get best action
                qValues = model.predict(board.vector.reshape(1,-1), verbose=0)[0]
                for action in np.argsort(qValues)[::-1]:
                    if board.validMove(*divmod(action, 3)):
                        break

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
        elif board.gameWon() == 2:
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



'''launches the user play function in curses wrapper'''
def playUser(model):
    curses.wrapper(__runUserPlay, model)



# ============= MAIN ====================================================================================
if __name__ == "__main__":
    '''m1 = loadModel('6.0.0', 0.01, 0.9, 15000)
    m2 = loadModel('6.1.0', 0.01, 0.9, 2500)
    m3 = loadModel('6.1.1', 0.01, 0.99, 10000)
    playUser(m3)'''
    trainModel()