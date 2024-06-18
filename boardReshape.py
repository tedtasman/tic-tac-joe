"""
06/03/2024
@authors: (Managing Director) Benjamin Rodgers and (CEO) Theodore Tasman

This is the board class for Tic Tac Toe. It describes the making of the board (3x3 grid) and the rules to play it.
This has methods that flip the turns between X & O, if the game is one or tied, and if the user made an error. 

Godspeed.

"""

import curses
import numpy as np


X = 1
O = -1

class Board:

    def __init__(self):

        self.grid = [[0,0,0],[0,0,0],[0,0,0]]
        self.nextMove = X
    
    
    # toggles move between X and O
    def __flipTurn(self):

        if self.nextMove == X:
            self.nextMove = O
        else:
            self.nextMove = X


    # returns token corresponding to value (1 -> X, -1 -> O, ? -> -)
    def decodeMove(self, move):

        if move == 1:
            return 'X'
        elif move == -1:
            return 'O'
        else:
            return ' '


    # get 1x9 vector representation of board
    @property
    def vector(self):
    
        vectorInput = np.zeros(9) #returns array of given shape (9) and its filled with zeroes

        # iterate through rows & cols of board
        for row in range(3):

            for col in range(3):
                
                vectorInput[row * 3 + col] = self.grid[row][col] # update vector with current grid value

        return vectorInput


    def validMove(self, row, col):

        if row < 0 or row > 2 or col < 0 or col > 2:
            # raise ValueError('Invalid row or column, try a proper number')
            return False
        
        elif self.grid[row][col] != 0:
            # raise ValueError('Position ({row}, {coulmn}) already occupied, try a differnet location.')
            return False
        
        else:
            return True

    #inserts current player's token to given location on grid
    def playMove(self, row, col):

        if self.validMove(row, col):
            self.grid[row][col] = self.nextMove # update grid
            self.__flipTurn() # flip turn
        else:
            print('Invalid move, try again.')
            


    # determines if winner of the game, returns winner or none
    def gameWon(self):

        #Check for row winner
        for row in self.grid:
            if (all(x == row[0] for x in row) and row[0] != 0) :
                return row[0]
            
        #Check for column winner 
        for col in range(3):
            if self.grid[0][col] == self.grid[1][col] == self.grid[2][col] and self.grid[0][col] != 0:
                return self.grid[0][col]
        
        #Check diagonal winner 
        if self.grid[0][0] == self.grid[1][1] == self.grid[2][2] and self.grid[0][0] != 0:
            return self.grid[0][0]
        elif self.grid[0][2] == self.grid[1][1] == self.grid[2][0] and self.grid[0][2] != 0:
            return self.grid[0][2]
        
        #Check for tie?
        elif all(all(x != 0 for x in row) for row in self.grid):
            return 2
        # if 3 in a row return winner, else None?

        else: #if no winner and game not over
            return 0
        
    
    # draw pretty board for in-place rendering
    def drawBoard(self, stdscr, rowHighlighted=None, colHighlighted=None):
        
        stdscr.clear()

        # iterate through all squares
        for row in range(3):

            for col in range(3):

                # if row matches selected and col not selected yet
                if row == rowHighlighted and colHighlighted==None:
                    # print row in highlight
                    stdscr.addstr(row * 2, col * 4, ' {} '.format(self.decodeMove(self.grid[row][col])), curses.A_REVERSE)

                # if both row and column match
                elif  row == rowHighlighted and col == colHighlighted:
                    # print spot in highlight
                    stdscr.addstr(row * 2, col * 4, ' {} '.format(self.decodeMove(self.grid[row][col])), curses.A_REVERSE)

                else:
                    # print normal
                    stdscr.addstr(row * 2, col * 4, ' {} '.format(self.decodeMove(self.grid[row][col])))

                # add vertical bars
                if col < 2:
                    stdscr.addstr(row * 2, col * 4 + 3, '|')
            
            # add horizontal bars
            if row < 2:
                stdscr.addstr(row * 2 + 1, 0, '---+---+---')
        
        stdscr.refresh()


    
    def __str__(self):
        out = '\n'
        for row in self.grid:
            for i in row:
                out += ' ' + self.decodeMove(i) + ' '
            out += '\n'
        return out
    
    __repr__ = __str__


