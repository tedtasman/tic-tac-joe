"""
06/05/2024
@authors: Benjamin Rodgers and (CEO) Theodore Tasman

This is the main AI training file. It builds the model, and iteratively updates it depending on the game results.

"""

import board as bd
import inOut as io
import numpy as np
import tensorflow as tf
import time


def __buildModel():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(9,)),  # Input layer (9 cells in Tic Tac Toe)
        tf.keras.layers.Dense(64, activation='relu'),  # Hidden layer
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
    

# run training
model = __buildModel()

for i in range(2):  # Play 1,000 games
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
            print(f'Iteration {i} ended in {j + 1} moves with result {result}')
            break

again = ''

while True:  # Play games until the user decides to quit
    
    # Ask the user if they want to play
    play = 'foo'
    # repeat until answered
    while True:
        # if invalid answer
        if play.lower() != "yes" and play.lower() != "y" and play.lower() != "no" and play.lower() != "n":
               # ask again
               play = input(f"\nDo you want to play {again}? (yes/no) ")
        else:
            break

    if play.lower() == "no" or play.lower() == "n":
        break

    board = bd.Board()
    inOut = io.InOut(board)
    while board.gameWon() == 0:
        if board.nextMove == 1:  # AI's turn
            print("\nJoe's Move:")
            time.sleep(1)
            row, col = getAction(board, model)
            board.playMove(row, col)
            print(board)
        else:  # Human's turn
            row,col = inOut.retrieveInput()
            board.playMove(row, col)
            print(board)

    # Print the result of the game
    if board.gameWon() == 1:
        print("Joe wins!")
    elif board.gameWon() == 2:
        print("You win!")
    else:
        print("It's a draw!")

    again = 'again'
