import json
import os
import numpy as np
from decorator import __init__

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

    def __init__(self, val_data):
        self.val_data = val_data

    def on_train_begin(self, logs={}):
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []

    def on_epoch_end(self, epoch, logs={}):
        # 5.4.1 For each validation batch
        predicted = []
        trueClasses = []
        for i in range(len(self.val_data)):
            data = self.val_data[i]
            r = self.model.predict_classes(data[0])
            r = r.flatten()
            predicted.extend(r)
            trueClasses.extend(data[1])

        val_f1 = round(f1_score(trueClasses, predicted), 4)
        val_recall = round(recall_score(trueClasses, predicted), 4)
        val_precis = round(precision_score(trueClasses, predicted), 4)

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