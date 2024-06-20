import random as rd
import numpy as np

TOPEDGE = 1
BOTTOMEDGE = 7
LEFTEDGE = 3
RIGHTEDGE = 5
TOPLEFT = 0
TOPRIGHT = 2
BOTTOMLEFT = 6
BOTTOMRIGHT = 8
CENTER = 4

'''
determines which strategy to play based on the input
'''
def trainerSwitch(strategy, board, actions):

    # play smart random strategy
    if strategy == 0:
        return smartRandom(board)
    
    # play L shape strategy
    elif strategy == 1:
        return playLShape(board, actions)
    
    # play corners strategy
    elif strategy == 2:
        return playCorners(board, actions)


'''
plays the L shape strategy for the first three moves in a random order
 X | X | O
---+---+---
   | X | 
---+---+---
 O |   |  
'''
def playLShape(board, actions):

    moves = []

    # if first move
    if board.nextMove not in board.vector:

        # play random square
        moves = [0, 1, 2, 3, 4, 5, 6, 7, 8]
        action = randomMove(board, moves)

    # if second move
    elif np.count_nonzero(board.vector == board.nextMove) == 1:
        previousMove = actions[-2]
        if previousMove in [TOPEDGE, BOTTOMEDGE]:
            moves = [previousMove + 1, previousMove - 1, CENTER]

        elif previousMove in [LEFTEDGE, RIGHTEDGE]:
            moves = [previousMove + 3, previousMove - 3, CENTER]
            
        elif previousMove == TOPLEFT:
            moves = [1, 3, CENTER]

        elif previousMove == TOPRIGHT:
            moves = [1, 5, CENTER]

        elif previousMove == BOTTOMLEFT:
            moves = [7, 3, CENTER]
        
        elif previousMove == BOTTOMRIGHT:
            moves = [7, 5, CENTER]

        else:
            moves = [0, 1, 2, 3, 5, 6, 7, 8]
            
        action = randomMove(board, moves)

    # if third move
    elif np.count_nonzero(board.vector == board.nextMove) == 2:
        
        previousMoves = (actions[-2], actions[-4])
        
        # if center wasn't played in the first two moves
        if CENTER not in previousMoves:

            # try to play center
            moves = [CENTER]

        elif TOPLEFT in previousMoves:

            # try to play adjacent edge
            moves = [1, 3]
        
        elif TOPRIGHT in previousMoves:

            # try to play adjacent edge
            moves = [1, 5]
        
        elif BOTTOMLEFT in previousMoves:

            # try to play adjacent edge
            moves = [7, 3]

        elif BOTTOMRIGHT in previousMoves:
                
            # try to play adjacent edge
            moves = [7, 5]
        
        elif TOPEDGE in previousMoves:

            # try to play adjacent corner
            moves = [0, 2]

        elif BOTTOMEDGE in previousMoves:

            # try to play adjacent corner
            moves = [6, 8]
        
        elif LEFTEDGE in previousMoves:

            # try to play adjacent corner
            moves = [0, 6]
        
        elif RIGHTEDGE in previousMoves:

            # try to play adjacent corner
            moves = [2, 8]

        action = randomMove(board, moves)

    # for all other moves
    else:
        action = smartRandom(board)

    return action


'''
plays the corners strategy for the first three moves in a random order
 X |   | O
---+---+---
   | O | 
---+---+---
 X |   | X 
'''
def playCorners(board, actions):

    moves = []

    # if first move
    if board.nextMove not in board.vector:

        # play random square
        moves = [0, 2, 6, 8]
        action = randomMove(board, moves)

    # if second move
    elif np.count_nonzero(board.vector == board.nextMove) == 1:

        # play a differnet corner
        previousMove = actions[-2]
        moves = [x for x in [0, 2, 6, 8] if x != previousMove]
        action = randomMove(board, moves)

    # if third move
    elif np.count_nonzero(board.vector == board.nextMove) == 2:
        
        # play a third corner
        previousMoves = (actions[-2], actions[-4])
        moves = [x for x in [0, 2, 6, 8] if x not in previousMoves]
        action = randomMove(board, moves)

    # for all other moves
    else:
        action = smartRandom(board)

    return action


'''
returns a random move from a list of moves.
if no valid moves are found, return a random move
'''
def randomMove(board, moves):

    if len(moves) > 0:
        # randomize the order of moves
        recommendedMoves = rd.sample(moves, len(moves))

        # iterate through the randomized moves
        for action in recommendedMoves: 

            # if the move is valid, return it
            if board.vector[action] == 0:
                return action
    
    # if no valid recommended moves are found, return a random move
    validMoves = [i for i in range(9) if board.vector[i] == 0]
    return rd.choice(validMoves)


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
finds all valid moves. if a winning move is found, return it. 
otherwise, return a random move
'''
def smartRandom(board):

    # find all valid moves
    validMoves = [i for i in range(9) if board.vector[i] == 0]

    # iterate through the valid moves
    for action in validMoves:

        # create a future board state with the move
        futureVector = board.vector.copy()
        futureVector[action] = board.nextMove

        # if this move will win the game, return it
        if vectorWin(futureVector) == board.nextMove:
            return action
    
    # if no winning moves are found, return a random move
    return rd.choice(validMoves)
