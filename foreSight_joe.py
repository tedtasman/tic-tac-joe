"""
06/11/2024
@authors: (Managing Director) Benjamin Rodgers and (CEO) Theodore Tasman

This is the main AI training file. It builds the model, and iteratively updates it depending on the game results.

"""

import numpy as np
import tensorflow as tf
import boardReshape as bd
import dualModel as dm

# get best action
def bestValidAction(vectorInput, model):

    # get qValues
    qValues = model.predict(vectorInput)[0]

    # sort qValues
    sortedIndices = np.argsort(qValues)[::-1]

    for action in sortedIndices: 

        # if current action is valid
        if vectorInput[action] == 0:

            return action # return current qValue index
        
    return -1

# predict future board states
def predictFutureStates(vectorInput, nextMove):

    futureStates = []

    # for every space on the board
    for i in range(9):

        # if other player can play there
        if vectorInput[i] == 0:

            # play that move on a copy of the board
            futureVector = vectorInput.copy()
            futureVector[i] = nextMove

            print(futureVector)
            # add the future state to the list
            futureStates.append(futureVector)

    return futureStates


# check immediate loss
def checkImmediateEnd(futureStates, nextMove):

    # for every future state
    for future in futureStates:

        # if the future state is a loss
        if vectorWin(future) == -nextMove:
            return -nextMove
        
        # if the future state is a win
        elif vectorWin(future) == nextMove:
            return nextMove
        
        # if the future state is a tie
        elif vectorWin(future) == 2:
            return 2

    # if no immediate end
    return 0

# check for win given vector state
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


# backpropagate


# train the model
def trainModel():

    pass




# ============= MAIN ====================================================================================
if __name__ == "__main__":
    a = dm.loadModel('v5.0.0-i5000')
    b = bd.Board()
    predictFutureStates(b.vector, b.nextMove)