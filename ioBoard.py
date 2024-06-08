"""
06/07/2024
@authors: (Managing Director) Benjamin Rodgers and (CEO) Theodore Tasman

This is the input output class for the Tic Tac Toe Game. 
The class will take the input from the user and can run methods based on what
the user has inputted. This class supercedes a previous version relating to I/O.
The difference between this and the old is that this cosmetically looks better as well as 
contains wrapping features to increase speed. 

"""






import curses
import time
import board as bd


def getMove(stdscr, board, user=None):
        
        # Display player's turn message and prompt for row input
        if user:
            stdscr.addstr(7, 0, "Your turn. Enter row (1, 2, or 3): ")
        else:
            stdscr.addstr(7, 0, "Player {}'s turn. Enter row (1, 2, or 3): ".format(board.decodeMove(board.nextMove)))
        row = int(stdscr.getkey()) - 1

        # Loop until a valid row is selected
        while True:
            if row not in range(3):
                stdscr.addstr(7, 0, "Invalid move. Enter row (1, 2, or 3): ")
                board.drawBoard(stdscr)
            elif 0 not in board.grid[row]:
                stdscr.addstr(7, 0, "That row is full. Enter a different row: ")
                board.drawBoard(stdscr)
            else:
                break
            row = int(stdscr.getkey()) - 1
                
        
        # Highlight the selected row
        row_highlighted = row
        board.drawBoard(stdscr, row_highlighted)

        # Loop until a valid column is selected
        while True:
            
            stdscr.addstr(8, 0, 'Enter column (1, 2, or 3): ')
            col = int(stdscr.getkey()) - 1
            # Loop until a valid column is selected
            while True:
                
                if col not in range(3):
                    stdscr.addstr(7, 0, "Invalid move. Enter col (1, 2, or 3): ")
                    board.drawBoard(stdscr)
                else:
                    break
                col = int(stdscr.getkey()) - 1

            if board.validMove(row, col):
                # Highlight the selected column
                col_highlighted = col
                board.drawBoard(stdscr, row_highlighted, col_highlighted)

                # Play the move and briefly highlight it
                board.playMove(row, col)
                board.drawBoard(stdscr, row_highlighted, col_highlighted)
                time.sleep(0.2)  # Pause for 0.5 seconds to highlight the move
                board.drawBoard(stdscr)
                break
            
            else:
                stdscr.addstr(7, 0, "Invalid move, try again. Enter row (1, 2, or 3): ")
                row = int(stdscr.getkey()) - 1
                board.drawBoard(stdscr)
                




# interactive UI with curses.py (superceeds runFree())
def runGame(stdscr):
    
    board = bd.Board()

    curses.curs_set(0) # hide the cursor
    
    while True:

        board.drawBoard(stdscr)

        getMove(stdscr, board)

        winner = board.gameWon() # check winner

        # if there was a winner
        if winner != 0 and winner != 3:
            board.drawBoard(stdscr) # redraw board
            stdscr.addstr(10, 0, "Player {} wins!".format(winner))  # Display the winning message
            stdscr.refresh()  # Refresh the screen to show the winning message
            stdscr.getkey()  # Wait for a key press before exiting
            break  # Exit the game loop


# launch interactive UI
def playGame():
    curses.wrapper(runGame)

if __name__ == "__main__":
    playGame()