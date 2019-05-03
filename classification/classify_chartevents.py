import csv
import json
import os
import pickle

from classification import chartevents_features

import keras

from keras.callbacks import ModelCheckpoint
from sklearn.model_selection._split import StratifiedKFold

from data_generators import LongitudinalDataGenerator
from keras_callbacks import SaveModelEpoch
from model_creators import MultilayerKerasRecurrentNNCreator
from metrics import f1, precision, recall

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
DATETIME_PATTERN = "%Y-%m-%d %H:%M:%S"

parametersFilePath = "parameters/classify_chartevents_parameters.json"

#Loading parameters file
print("========= Loading Parameters")
parameters = None
with open(parametersFilePath, 'r') as parametersFileHandler:
    parameters = json.load(parametersFileHandler)
if parameters is None:
    exit(1)

if not os.path.exists(parameters['modelCheckpointPath']):
    os.mkdir(parameters['modelCheckpointPath'])

data = np.array([parameters["allDataPath"] + x for x in os.listdir(parameters["allDataPath"])])
# Get label for stratified k-fold
allDataGenerator = LongitudinalDataGenerator(data, 1)
labelsForStratified = []
for d in range(len(allDataGenerator)):
    labelsForStratified.append(allDataGenerator[d][1][0][0])

print("========= Preprocessing data")
#TODO : preprocessing for classification

kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=15)
inputShape = (parameters['dataLength'], len(chartevents_features.FEATURES_ITEMS_LABELS.keys()))

config = None
if os.path.exists(parameters['modelConfigPath']):
    with open(parameters['modelConfigPath'], 'r') as configHandler:
        config = json.load(configHandler)

i = 0
# ====================== Script that start training new models
with open(parameters['resultFilePath'], 'a+') as cvsFileHandler: # where the results for each fold are appended
    dictWriter = None
    for trainIndex, testIndex in kf.split(data, labelsForStratified):
        print(len(trainIndex))
        if config is not None and config['fold'] > i:
            print("Pass fold {}".format(i))
            i += 1
            continue
        if config is not None and config['epoch'] == parameters['trainingEpochs']:
            print("Pass fold {}-".format(i))
            i += 1
            continue
        # Training an instance of Word2Vec model with the training data
        print("======== Fold {} ========".format(i))

        # If exists a valid config  to resume a training
        if config is not None and config['fold'] == i and config['epoch'] < parameters['trainingEpochs']:
            epochs = parameters['trainingEpochs'] - config['epoch']

            print("========= Loading generators")
            with open(parameters['trainingGeneratorPath'], 'rb') as trainingGeneratorHandler:
                dataTrainGenerator = pickle.load(trainingGeneratorHandler)

            with open(parameters['testingGeneratorPath'], 'rb') as testingGeneratorHandler:
                dataTestGenerator = pickle.load(testingGeneratorHandler)

            kerasAdapter = MultilayerKerasRecurrentNNCreator.create_from_path(config['filepath'],
                                                    custom_objects={'f1':f1, 'precision':precision, 'recall':recall})
            configSaver = SaveModelEpoch(parameters['modelConfigPath'],
                                         parameters['modelCheckpointPath'] + 'fold_' + str(i), i, alreadyTrainedEpochs=config['epoch'])
        else:
            dataTrainGenerator = LongitudinalDataGenerator(data[trainIndex], 1)
            dataTestGenerator = LongitudinalDataGenerator(data[testIndex], 1)
            print("========= Saving generators")
            with open(parameters['trainingGeneratorPath'], 'wb') as trainingGeneratorHandler:
                pickle.dump(dataTrainGenerator, trainingGeneratorHandler, pickle.HIGHEST_PROTOCOL)

            with open(parameters['testingGeneratorPath'], 'wb') as testingGeneratorHandler:
                pickle.dump(dataTestGenerator, testingGeneratorHandler, pickle.HIGHEST_PROTOCOL)

            modelCreator = MultilayerKerasRecurrentNNCreator(inputShape, parameters['outputUnits'], parameters['numOutputNeurons'],
                                                             loss=parameters['loss'], layersActivations=parameters['layersActivations'],
                                                             gru=parameters['gru'], use_dropout=parameters['useDropout'],
                                                             dropout=parameters['dropout'],
                                                             metrics=[f1, precision, recall, keras.metrics.binary_accuracy])
            kerasAdapter = modelCreator.create()
            epochs = parameters['trainingEpochs']
            configSaver = SaveModelEpoch(parameters['modelConfigPath'],
                                         parameters['modelCheckpointPath'] + 'fold_' + str(i), i)

        modelCheckpoint = ModelCheckpoint(parameters['modelCheckpointPath']+'fold_'+str(i))
        kerasAdapter.fit(dataTrainGenerator, epochs=epochs, batch_size=len(dataTrainGenerator),
                         validationDataGenerator=dataTestGenerator, validationSteps=len(dataTestGenerator),
                         callbacks=[modelCheckpoint, configSaver])
        result = kerasAdapter.evaluate(dataTestGenerator, batch_size=len(dataTestGenerator))
        result["fold"] = i
        if dictWriter is None:
            dictWriter = csv.DictWriter(cvsFileHandler, result.keys())
        if config['fold'] == 1:
            dictWriter.writeheader()
        dictWriter.writerow(result)
        dataTrainGenerator.clean_files()
        dataTestGenerator.clean_files()
        i += 1
