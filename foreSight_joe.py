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
import boardReshape as bd
import random as rd
import ioBoard as io
import tensorflow as tf
import matplotlib.pyplot as plt

BUILD = '6.1.0'



# ======== HYPERPARAMETERS ===========

ALPHA = 0.01     # learning rate
GAMMA = 0.9     # discount factor

EPSILON = 1.0   # exploration rate
EPSILON_DECAY = 0.995 # decay rate for epsilon
EPSILON_MIN = 0.1 # minimum epsilon

TIE_REWARD = 0.5  # reward for tie
BLUNDER_PENALTY = -1 # penalty for blunder (missed win or unforced loss)



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
        tf.keras.layers.Dense(9, activation='softmax')  # Output layer (9 possible actions)
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(alpha), loss='categorical_crossentropy')

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
def loadModel(build, alpha, gamma, iteration):
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
        if sum(vector[0::4]) == 3 or sum(vector[2:7:2]) == 3:
            return 1
    
        elif sum(vector[0::4]) == -3 or sum(vector[2:7:2]) == -3:
            return -1

        # if all spaces filled (tie)
        if 0 not in vector:
            return 2

        return 0



'''
Gets best moves for exploitation, sets blunder penalty for missed wins and unforced losses

returns target qValues
'''
def exploit(vectorInput, currentModel, board, blunderPenalty, tieReward):

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
    for i in range(9):
        if i not in validActions:
            qValues[i] = float('-inf')

    # get best action
    action = np.argmax(qValues)

    # play move
    board.playMove(*divmod(action, 3))

    # return adjusted qValues
    return qValues, action



'''
Checks for a win or tie available. If none, plays a random move.
If win, sets qValue of action to 1. If tie, sets qValue of action to tie reward.

returns targetQValues
'''
def explore(vectorInput, currentModel, board, tieReward, random=False):
    
    # get qValues based on current turn
    qValues = currentModel.predict(vectorInput.reshape(1,-1), verbose=0)[0]

    # get valid indices
    validActions =  [i for i in range(9) if board.validMove(*divmod(i, 3))]


    # set smart to False to indicate random move
    smart = False

    # loop through best actions to check for win or tie
    for action in validActions:
        
        # play move on copy of board
        futureVector = vectorInput.copy()
        futureVector[action] = board.nextMove

        # if current agent wins with this move
        if vectorWin(futureVector) == board.nextMove:

            qValues[action] = 1
            smart = True # set smart to True to indicate a smart move can be made

        # if current agent ties with this move
        elif vectorWin(futureVector) == 2:

            qValues[action] = tieReward
            smart = True # set smart to True to indicate a smart move can be made

    # set all invalid moves to negative infinity
    for i in range(9):
        if i not in validActions:
            qValues[i] = float('-inf')

    # if there is a smart move
    if smart:
        # get best action - will be the smart move
        action = np.argmax(qValues)
        row, col = divmod(action, 3)
    
    # if no smart move
    else:

        # get random valid action
        action = rd.choice(validActions)
        row, col = divmod(action, 3)

    # play move
    board.playMove(row, col)
    
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


'''
train the model using the hyperparameters defined above or custom hyperparameters if override is set to True.

returns dualModel object
'''
def trainModel(override = False):

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

    # define training
    iterations = int(input("Set iterations: "))
    saveInterval = int(input("Set save interval: "))
    print('\n====================================================\n')
    print('Running {} iterations with save interval {}... (ctrl + C to exit)\n'.format(iterations, saveInterval))

    # get model
    model = __buildModel(alpha)

    wins = 0
    losses = 0
    ties = 0
    winPercentage = 0
    winPercentages = []
    averageMoves = 0

    # run training
    for i in range(iterations):
            
        # create new board
        board = bd.Board()

        # get vector representation of empty board
        vectorInput = np.zeros(9)

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
        print(f"\033[1K\r\033[0KAfter {i + 1} iterations: \t W-L-T: {wins}-{losses}-{ties} \t Win/Loss: {(wins + 0.5 * ties) / (i + 1):.3f} \t Average Moves: {averageMoves:.1f}", end='')

        if (i + 1) % saveInterval == 0:
            print('\n')
            saveModel(model, BUILD, alpha, gamma, i + 1)
            print('\n')

    print('\n\nTraining complete!\n')

    # plot win percentage over time
    plt.plot(winPercentages)
    plt.title('Win Percentage Over Time')
    plt.xlabel('Iteration')
    plt.ylabel('Win Percentage')
    plt.show()

    return model


# runs AI vs user game in curses wrapper
def __runUserPlay(stdscr, dualModel):

    """
    This is a function that allows the user to play against the model that has been trained.
    Very straight forward.
    """
    
    play = 'foo'
    
    while True:  # Play games until the user decides to quit

        board = bd.Board()
        joeTurn = rd.randint(-1,1)
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
                action = dualModel.bestValidAction(board)

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



# launches the user play function in curses wrapper
def playUser(model):
    curses.wrapper(__runUserPlay, model)



# plays two models against each other
def playModels(dualModel1, dualModel2, verbose=False, games=None):

    # repeat until valid answer
    while not isinstance(games, int) or int(games) < 1:
        games = int(input("\nHow many games to play? "))
            
    wins = 0
    ties = 0

    for i in range(games): 

        # randomly decide first move
        model1Turn = rd.randint(1,2)
        # make new board
        board = bd.Board()

        if model1Turn == 1:
            model1 = dualModel1.xModel
            model2 = dualModel2.oModel
        else:
            model1 = dualModel1.oModel
            model2 = dualModel2.xModel

        # until a winner is decided
        while board.gameWon() == 0:

            # get current state
            vectorInput = board.vector

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

        if verbose:
            print(board)

        # Update the win count
        winner = board.gameWon()
        if winner == model1Turn:
            wins += 1
        elif winner == 3:
            ties += 1
        

        print("\033[2K\r{} STATS: \t W-L-T: {}-{}-{} \t Games Played: {} \t Win/Loss: {:0.3f}".format(dualModel1.name, wins, (i - wins - ties + 1), ties, (i + 1), (wins + 0.5 * ties) / (i + 1)), end='')

    print('\n')
    return (wins + 0.5 * ties) / (i + 1) 




# ============= MAIN ====================================================================================
if __name__ == "__main__":
    trainModel()