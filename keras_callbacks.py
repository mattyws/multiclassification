import json
import os

from keras.callbacks import Callback

class SaveModelEpoch(Callback):

    """
    Save a configuration file wth the last epoch trained and the path of the model file.
    This configuration file will be used to resume the model training.
    """

    def __init__(self, configFilePath, modelFilePath, foldNumber, alreadyTrainedEpochs=0):
        self.configFilePath = configFilePath
        self.modelFilePath = modelFilePath
        self.foldNumber = foldNumber
        self.alreadyTrainedEpochs = alreadyTrainedEpochs

    def on_epoch_end(self, epoch, logs={}):
        with open(self.configFilePath, 'w') as configFileHandler:
            json.dump({"epoch": (epoch+1)+self.alreadyTrainedEpochs, "filepath":self.modelFilePath,
                       "fold":self.foldNumber}, configFileHandler)
        return