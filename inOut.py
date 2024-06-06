"""

06/04/2024
@authors: Benjamin Rodgers and (CEO) Theodore Tasman

This is the input output class for the Tic Tac Toe Game. 
The class will take the input from the user and can run methods based on what
the user has inputted. Other methods are to be implemented in the future when the AI
is developed.

"""

class InOut:
    
    def __init__(self,board):
        self.board = board
    

    def retrieveInput(self):
        
        """
        
        This method will recieve the input from the user and 
        store it in the row and column variables.
        
        """
      
        print('Enter Your move:\n')
        
        while True: # loop until valid input

          row = int(input("Enter row: ")) - 1
          col = int(input("Enter column: ")) - 1

          if row < 0 or row > 2 or col < 0 or col > 2 or row == '' or col == '' : # if inputs out of range
            print('Invalid row or column, try again.')

          elif self.board.grid[row][col] != 0: # if inputs occupied
            print('The position of ({}, {}) is already occupied, try a unused space.'.format(row+1, col+1))
      
          else: # if all passed
             break # escape
      
        return row, col


    def runFree(self):
        
        """
        
        This method is underdeveloped. The goal is for the AI to use this to know
        when to stop playing. The AI has not been developed yet, so this method is
        not really needed just yet. 
        
        """

        # main game loop
        while self.board.gameWon() == 0:
            row, col = self.retrieveInput()
            self.board.playMove(row,col)
            print(self.board)
        
        # result output
        outcome = self.board.gameWon()
        if outcome == 1:
            print('Player X wins.')
        elif outcome == 2:
            print('Player O wins.')
        else:
           print("It's a draw.")
