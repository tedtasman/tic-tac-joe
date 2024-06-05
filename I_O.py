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
        row = int(input("Enter row: "))
        col = int(input("Enter column: "))
        
        #Error checking for invalid row or column & checking for occupied space
        while (row < 0 or row > 2 or col < 0 or col > 2) or (self.board.grid[row][col] != 0):
          if row < 0 or row > 2 or col < 0 or col > 2:
            print('Invalid row or column, try again')
            row = int(input("Enter row: "))
            col = int(input("Enter column: "))
          elif self.board.grid[row][col] != 0:
            print('The position of ({row}, {col}) is already occupied, try a unused space.')
            row = int(input("Enter row: "))
            col = int(input("Enter column: "))
      
        
        #Return the row and column
        return row, col
    
    def moveMaker(self,row,col):
        """

        This method takes the users inputs and plays them on the board. 

        """

        if row and col != None:
          self.board.playMove(row,col)

    def runFree(self):
        
        """
        
        This method is underdeveloped. The goal is for the AI to use this to know
        when to stop playing. The AI has not been developed yet, so this method is
        not really needed just yet. 
        
        """

        while not self.board.gameWon():
            row, col = self.retrieveInput()
            self.moveMaker(row,col)
            print(self.board)
