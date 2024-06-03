"""
06/03/2024
@authors: Benjamin Rodgers and (CEO) Theodore Tasman

This is the board class for Tic Tac Toe. It describes the making of the board (3x3 grid) and the rules to play it.
This has methods that flip the turns between X & O, if the game is one or tied, and if the user made an error. 

Godspeed.

"""




X = 1
O = 2

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


    # returns token corresponding to value (1 -> X, 2 -> O, ? -> -)
    def __decodeMove(self, move):

        if move == 1:
            return 'X'
        elif move == 2:
            return 'O'
        else:
            return '-'


    #inserts current player's token to given location on grid
    def playMove(self, row, col):

        if row < 0 or row > 2 or col < 0 or col > 2:
            raise ValueError('Invalid row or column, try a proper number')
        
        if self.grid[row][col] != 0:
            raise ValueError('Position ({row}, {coulmn}) already occupied, try a differnet location.')

        self.grid[row][col] = self.nextMove # update grid

        self.__flipTurn() # flip turn


    # determines if winner of the game, returns winner or none
    def gameWon(self):

        #Check for row winner
        for row in self.grid:
            if (all(x == row[0] for x in row) and row[0] != 0) or (all(O == row[0] for O in row) and row[0] != 0) :
                return row[0]
            
        #Check for column winner 
        for col in range(3):
            if self.grid[0][col] == self.grid[1][col] == self.grid[2][col] and self.grid[0][col] != 0:
                return self.grid[0][col]
        
        #Check diagonal winner 
        if self.grid[0][0] == self.grid[1][1] == self.grid[2][2] and self.grid[0][0] != 0:
            return self.grid[0][0]
        if self.grid[0][2] == self.grid[1][1] == self.grid[2][0] and self.grid[0][2] != 0:
            return self.grid[0][2]
        
        #Check for tie?
        if all(all(x != 0 for x in row) for row in self.grid):
            return 0
        # if 3 in a row return winner, else None?

    
    def __str__(self):
        out = '\n'
        for row in self.grid:
            for i in row:
                out += ' ' + self.__decodeMove(i) + ' '
            out += '\n'
        return out
    
    __repr__ = __str__