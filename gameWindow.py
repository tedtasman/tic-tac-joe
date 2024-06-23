import tkinter as tk
import board as bd
import joePlayer as joe
import random as rd

# Constants
X = 1
O = -1
XColor = "#9999FF"
OColor = "#FF9999"
TIEColor = "#99FF99"
BG = "#333333"
FG = "#EEEEEE"
MID = "#AAAAAA"

joeQuotes = ["What do we have here...?", "I am inevitable...", "Hmmm...", "I am Joe!", "Your time is up!", "I've got you now!", "Watch out!", "You can't beat me!", "I'm in control now!", "I'm always one step ahead!", "You're finished!"]
joeTies = ["Well played...", "We're evenly matched...", "Well, that was boring...", "That was close!"]
joeWins = ["I win!", "I'm the best!", "I'm unbeatable!", "I'm the champion!", "I'm the king!"]
joeLosses = ["You win this time...", "You got lucky...", "I'll get you next time...", "I'll be back!", "I am just an average Joe..."]

# Globals
board = bd.Board()
currentPlayer = X
joeTurn = O
gameOver = False

# Create the main window
window = tk.Tk()
window.title("Tic-Tac-Toe")
window.configure(background=BG)


# Create a label for the quote
quoteLabel = tk.Label(window, text="", background=BG, foreground=FG, font=("Helvetica", 16)) 
quoteLabel.pack(pady=20)


def updateQuoteLabel():
    if currentPlayer == joeTurn:
        quoteLabel.config(text=f"Joe: {rd.choice(joeQuotes)}")



# Create a frame for the Tic-Tac-Toe board
boardFrame = tk.Frame(window, background=MID)
boardFrame.pack(pady=20, padx=20)


# Function to handle button click
def handleClick(row, col):

    global currentPlayer, gameOver

    if gameOver:
        resetGame()

    elif board.grid[row][col] == 0 and currentPlayer != joeTurn:

        # play user move
        board.playMove(row, col)
        buttons[row][col].config(text=board.decodeMove(currentPlayer), fg=XColor if currentPlayer == X else OColor)

        # switch player and update
        currentPlayer = board.nextMove
        updateStatusLabel()

        # Schedule Joe's move after a delay to allow UI to update
        if not gameOver:
            updateQuoteLabel()
            window.after(1500, joeMove)

def joeMove():

    global currentPlayer, gameOver

    if currentPlayer == joeTurn and not gameOver:

        row, col = joe.getJoeMove(board)
        if board.grid[row][col] == 0:

            board.playMove(row, col)
            buttons[row][col].config(text=board.decodeMove(currentPlayer), fg=XColor if currentPlayer == X else OColor)
            
            currentPlayer = board.nextMove
            updateStatusLabel()


# Function to check for a win
def checkWin(player):
    return True if board.gameWon() == player else False


# Function to check for a draw
def checkDraw():
    return True if board.gameWon() == 2 else False


# Function to reset the game
def resetGame():

    global currentPlayer, board, gameOver

    gameOver = False
    board = bd.Board()

    for row in range(3):

        for col in range(3):

            buttons[row][col].config(text=" ")

    currentPlayer = X
    updateStatusLabel()


# Create the buttons within the board frame
buttons = []
for row in range(3):

    buttonRow = []

    for col in range(3):

        button = tk.Button(boardFrame, background=BG ,text=" ", width=3, height=1, font=("Helvetica", 50), relief=tk.FLAT, command=lambda row=row, col=col: handleClick(row, col))
        button.grid(row=row, column=col, padx=1, pady=1)
        buttonRow.append(button)

    buttons.append(buttonRow)


# determine who's turn it is (Joe or User)
def determinePlayer(turn):

    global joeTurn

    if joeTurn == turn:
        return "Joe"
    else:
        return "User"


# switch Joe's turn (first or second)
def switchJoeTurn():

    global joeTurn
    
    if joeTurn == X:
        joeTurn = O
    else:
        joeTurn = X
        window.after(1000, joeMove)
    
    resetGame()
    updateSwitchButton()
    updateQuoteLabel()


def updateSwitchButton():
    switchButton.config(text=f"Make {determinePlayer(O)} Go First", font=("Helvetica", 16))


def updateStatusLabel():

    global gameOver

    if checkDraw():
        gameOver = True
        statusLabel.config(text="It's a draw!", fg=TIEColor, font=("Helvetica", 16))
        quoteLabel.config(text=f"Joe: {rd.choice(joeTies)}", font=("Helvetica", 16))

    elif checkWin(currentPlayer * -1):
        gameOver = True
        statusLabel.config(text=f"{determinePlayer(currentPlayer * -1)} ({board.decodeMove(currentPlayer * -1)}) wins!", fg=XColor if currentPlayer == O else OColor, font=("Helvetica", 16))
        if currentPlayer == joeTurn:
            quoteLabel.config(text=f"Joe: {rd.choice(joeLosses)}")
        else:
            quoteLabel.config(text=f"Joe: {rd.choice(joeWins)}")

    else:
        statusLabel.config(text=f"{determinePlayer(currentPlayer)}'s Turn ({board.decodeMove(currentPlayer)})", fg=FG, font=("Helvetica", 16))


# Additional controls and information

statusLabel = tk.Label(window, bg=BG, fg=FG, text=f"{determinePlayer(currentPlayer)}'s Turn ({board.decodeMove(currentPlayer)})", font=("Helvetica", 16))
statusLabel.pack(pady=20)


switchPanel = tk.Frame(window)
switchPanel.config(bg=BG)
switchPanel.pack(pady=20)

resetButton = tk.Button(switchPanel, bg=BG, fg=FG, text="Reset Game", command=resetGame, font=("Helvetica", 12))
resetButton.grid(row=0, column=0, padx=5)  # Use grid with padx

switchButton = tk.Button(switchPanel, bg=BG, fg=FG, text=f"Make {determinePlayer(O)} Go First", command=switchJoeTurn, font=("Helvetica", 12))
switchButton.grid(row=0, column=1, padx=5)  # Use grid with padx







# Run the main loop
window.mainloop()