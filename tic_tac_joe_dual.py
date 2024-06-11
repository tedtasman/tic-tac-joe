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

build = '5.3.0'

# ======== HYPERPARAMETERS ===========

alpha = 0.01     # learning rate -> Reccomend increase
gamma = 0.9     # discount factor -> Reccomend increase
epsilon = 1.0   # exploration rate -> Not sure

epsilonDecay = 0.995
epsilonMin = 0.1

tieReward = 0

#============================

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


# determine winner
def winnerDeter(board):
    win = board.gameWon()
    if win == 1:
        return 1
    elif win == 2:
        return -1
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
        dualModel = dm.loadModel('v{}-i{}'.format(version, iteration))
        offset = iteration

    else:
        dualModel = dm.DualModel('joe')

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
                action = dualModel.bestValidAction(board, qValues)
                # convert to row, col
                row, col = divmod(action, 3)

            # play move
            board.playMove(row, col)

            newVectorInput = boardStateValue(board)
            # save move for backprop
            boardStates.append((newVectorInput, action))

            # check for winner
            result = winnerDeter(board)

            # set target to current probabilities
            target = qValues[:]

            # if there was a winner
            if result != 0:
                
                # reward based on result
                target[3 * row + col] = abs(result)

            # if current game was a tie
            elif board.gameWon() == 3:
                # give tieReward (hyperparameter)
                target[3 * row + col] = tieReward

            # retrain model
            currentModel.fit(vectorInput.reshape(1,-1), np.array([target]), verbose=0)
            
            # update board state
            vectorInput = newVectorInput

            # escape to next round if there was a winner
            if result != 0:
                break

        winner = board.gameWon()
        finalModel = currentModel

        # if O up next, O is other model
        if board.nextMove == 2:
            otherModel = dualModel.oModel
        else:
            # flip result to reward o model and punish x model
            otherModel = dualModel.xModel
            result *= -1

        # if game was a tie, set result to tieReward
        if result == 0:
            result = tieReward

        turn = 1
        for state, action in reversed(boardStates): 

            # if same model as final model 
            if turn % 2 == 1:
                currentModel = finalModel
            else:
                currentModel = otherModel
    
            # get qvalues for that state
            qValues = currentModel.predict(state.reshape(1,-1), verbose=0)[0]
            target = qValues[:]

            # update targer of action to final result * gamma
            target[action] = result * gamma
            
            # retrain current model
            currentModel.fit(vectorInput.reshape(1,-1), np.array([target]), verbose=0)

            # flip result for other model
            if winner != 3:
                result *= -gamma
            else:
                result *= gamma

            turn += 1 # track turns


        print("\rIteration {} ended in {} moves. {}!".format(i + offset + 1, j + 1, 'X wins' if winner == 1 else 'O wins' if winner == 2 else "A draw"), end='')
        
        if (i + offset + 1) % saveInterval == 0:
            dualModel.name = 'v{}-i{}'.format(build, (i + offset + 1)) #Not the best, but it works!
            dualModel.saveModel()

    return dualModel


# get best action (board, qValues) -> qvalue
def bestValidAction(board, qValues):

    # sort qValues
    sortedIndices = np.argsort(qValues)[::-1]

    for i in sortedIndices: # Highly unlikely, but qvalues could be a problem 
        
        # get row/column from qValue
        row, col = divmod(i, 3)

        # if current action is valid
        if board.validMove(row, col):

            return i # return current qValue index
        
    return -1


# runs AI vs user game in curses wrapper
def __runUserPlay(stdscr, dualModel):

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

        # select joe model to use
        if joeTurn == 1:
            model = dualModel.xModel
        else:
            model = dualModel.oModel

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


# launches the user play function in curses wrapper
def playUser(model):
    curses.wrapper(__runUserPlay, model)


# **NOT UPDATED FOR DUAL MODEL ARCHITECTURE** deletes all the models in the model directory 
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

        if verbose:
            print(board)

        # Update the win count
        winner = board.gameWon()
        if winner == model1Turn:
            wins += 1
        elif winner == 3:
            ties += 1
        

        print("\033[K{} STATS: \t W-L-T: {}-{}-{} \t Games Played: {} \t Win/Loss: {:0.3f}".format(dualModel1.name, wins, (i - wins - ties + 1), ties, (i + 1), (wins + 0.5 * ties) / (i + 1)), end='\r')

    print('\n')
    return (wins + 0.5 * ties) / (i + 1)


def runAutoTest(models, games=50):
    """
    This is the function that runs the autotester. It will run the playModels function on all the models in the models directory.
    """

    modelStats = {}

    # iterate through models
    for model1 in models:
        for model2 in models:

            # if different models
            if model1 != model2:

                model1Name = model1.name
                model2Name = model2.name

                # if model1 not in stats
                if model1Name not in modelStats:

                    # add model1 to stats
                    modelStats[model1Name] = {}

                if model2Name not in modelStats:
                        
                    # add model2 to stats
                    modelStats[model2Name] = {}

                # if model1 vs model 2 not run yet
                if model2Name not in modelStats[model1Name]:
                        
                        # run models
                        print ("{} vs {}".format(model1Name, model2Name))
                        modelStats[model1Name][model2Name] = playModels(model1, model2, games=games)
                        modelStats[model2Name][model1Name] = 1 - modelStats[model1Name][model2Name]

    for model1 in modelStats:
        print(model1, 'vs:')
        for model2 in modelStats[model1]:
            print("\t{}: {:0.3f}".format(model2, modelStats[model1][model2]))



# ============= MAIN ====================================================================================
if __name__ == "__main__":
    trainModel(build, 1000)
    '''a = dm.loadModel('v5.0.0-i1000')
    b = dm.loadModel('v5.1.0-i1000')
    c = dm.loadModel('v5.2.0-i1000')
    d = dm.loadModel('v5.3.0-i1000')'''
