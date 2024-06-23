import keras
import numpy as np

model = keras.models.load_model('6.5.6st/a0.01_g0.9_i10000.keras')

def getJoeMove(board):

    # get best action
    qValues = model.predict(board.vector.reshape(1,-1), verbose=0)[0]
    for action in np.argsort(qValues)[::-1]:
        if board.validMove(*divmod(action, 3)):
            break

    # return move
    row, col = divmod(action, 3)
    return row, col