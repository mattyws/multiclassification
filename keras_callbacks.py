import json
import os
import numpy as np

from keras.callbacks import Callback
from sklearn.metrics import f1_score, recall_score, precision_score


class ResumeTrainingCallback(Callback):

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


# Validation metrics callback: validation precision, recall and F1
# Some of the code was adapted from https://medium.com/@thongonary/how-to-compute-f1-score-for-each-epoch-in-keras-a1acd17715a2
class Metrics(Callback):

    def on_train_begin(self, logs={}):
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []

    def on_epoch_end(self, epoch, logs={}):
        # 5.4.1 For each validation batch
        for batch_index in range(0, len(self.validation_data)):
            # 5.4.1.1 Get the batch target values
            temp_targ = self.validation_data[batch_index][1]
            # 5.4.1.2 Get the batch prediction values
            temp_predict = self.model.predict_classes(self.validation_data[batch_index][0]).flatten()
            # 5.4.1.3 Append them to the corresponding output objects
            if(batch_index == 0):
                val_targ = temp_targ
                val_predict = temp_predict
            else:
                val_targ = np.vstack((val_targ, temp_targ))
                val_predict = np.vstack((val_predict, temp_predict))

        val_f1 = round(f1_score(val_targ, val_predict), 4)
        val_recall = round(recall_score(val_targ, val_predict), 4)
        val_precis = round(precision_score(val_targ, val_predict), 4)

        self.val_f1s.append(val_f1)
        self.val_recalls.append(val_recall)
        self.val_precisions.append(val_precis)

        # Add custom metrics to the logs, so that we can use them with
        # EarlyStop and csvLogger callbacks
        logs["val_f1"] = val_f1
        logs["val_recall"] = val_recall
        logs["val_precis"] = val_precis

        print("— val_f1: {} — val_precis: {} — val_recall {}".format(
                 val_f1, val_precis, val_recall))
        return