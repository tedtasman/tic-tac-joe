"""

06/08/2024
@authors: (Managing Director) Benjamin Rodgers and (CEO) Theodore Tasman

This class sets up two models, one as X and one as O. This allows the individual models to learn
the game and seperately and not get confused when trying to figure out if it is x or y.

"""
import os
import numpy as np 
import tensorflow as tf 
import foreSight_joe as joe

# ======== HYPERPARAMETERS ===========

ALPHA = joe.ALPHA     # learning rate


class DualModel:
    
    def __init__(self, name, alpha=ALPHA):
        self.name = name
        self.alpha = alpha
        self.xModel = self.__buildModel()
        self.oModel = self.__buildModel()



    def __buildModel(self):
        model = tf.keras.models.Sequential([
            tf.keras.layers.Input(shape=(9,)),  # Input layer (9 cells in Tic Tac Toe)
            tf.keras.layers.Dense(64, activation='relu'),  # Hidden layer
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(64, activation='relu'),  # Hidden layer 2 
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(64, activation='relu'),  # Hidden layer 3
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(9, activation='softmax')  # Output layer (9 possible actions)
        ])

        model.compile(optimizer=tf.keras.optimizers.Adam(self.alpha), loss='categorical_crossentropy')

        return model


    def bestValidAction(self, board):
    
        # determine whose move it is
        currentModel = self.getCurrentModel(board) 
        
        # get qValues based on current turn
        vectorInput = board.vector

        # get qValues from model
        qValues = currentModel.predict(vectorInput.reshape(1,-1), verbose=0)[0]

        #get valid indices
        validIndices =  [i for i in range(9) if board.validMove(*divmod(i, 3))]
            
        # filter qValues using valid indices
        validQValues = qValues[validIndices]
    
         # sort valid qValues
        sortedIndices = np.argsort(validQValues)[::-1]

        for i in sortedIndices:
            
            # get row/column from qValue
            row, col = divmod(i, 3)

            # if current action is valid
            if board.validMove(row, col):
                return i # return current qValue index
        
        # return -1 if no next move (game is a tie)
        return -1


    def getCurrentModel(self, board):

        if board.nextMove == 1:
            currentModel = self.xModel
        else:
            currentModel = self.oModel

        return currentModel


    def saveModel(self, build):
        """
        This saves a model to a desired path for later use.
        """
        
        # make 
        try:
            os.makedirs(f'./{build}/{self.name}')
        except FileExistsError:
            print("Hey pal, that model name is already in use...")
            return
        
        self.xModel.save(f"./{build}/{self.name}/X_{self.name}.keras") # save X model
        self.xModel.save(f"./{build}/{self.name}/O_{self.name}.keras") # save O model

        print("\nMODEL SAVED:\t./dualModels/{}/\n".format(self.name))

    

def loadModel(build, name):
    """
    This loads a model from a desired path for later use.
    """
    xModel = tf.keras.models.load_model(f"./{build}/{name}/X_{name}.keras") # load X model
    oModel = tf.keras.models.load_model(f"./{build}/{name}/O_{name}.keras") # load O model
    print(f"\nMODEL LOADED:\t./{build}/{name}/\n")

    dualModel = DualModel(name)
    dualModel.xModel = xModel
    dualModel.oModel = oModel

    return dualModel