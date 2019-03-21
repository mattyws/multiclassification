import csv
import json
import os
import pickle
from datetime import datetime

import pandas as pd
import chartevents_features

import keras

import gensim
import math

import numpy
from keras.callbacks import ModelCheckpoint
from nltk import regexp_tokenize
from sklearn.metrics.classification import f1_score
from sklearn.model_selection import train_test_split, KFold
from sklearn.model_selection._split import StratifiedKFold

from data_generators import Word2VecTextEmbeddingGenerator
from data_representation import Word2VecEmbeddingCreator
from keras_callbacks import SaveModelEpoch
from model_creators import MultilayerKerasRecurrentNNCreator
from metrics import f1, precision, recall

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
DATETIME_PATTERN = "%Y-%m-%d %H:%M:%S"


def data_transform(dataPath, data, labels, word2vecModel, embeddingSize, batchSize, maxWords=None):
    generator = Word2VecTextEmbeddingGenerator(dataPath, word2vecModel, batchSize, embeddingSize=embeddingSize,
                                               iterForever=True)
    for x, y in zip(data, labels):
        generator.add(x, y, maxWords=maxWords)
    return generator

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

# only load the dataset if do not resuming a training, use script paramters for that
if os.path.exists(parameters['modelCheckpointPath']+parameters['datasetFilesFileName']):
    print("========= Loading previous dataset")
    datasetFiles = []
    datasetLabels = []
    with open(parameters['modelCheckpointPath']+parameters['datasetFilesFileName'], 'rb') as datasetFilesHandler:
        datasetFiles = pickle.load(datasetFilesHandler)

    with open(parameters['modelCheckpointPath']+parameters['datasetLabelsFileName'], 'rb') as datasetLabelsHandler:
        datasetLabels = pickle.load(datasetLabelsHandler)
else:
    # Get files paths
    print("========= Getting files paths")
    datasetFiles = []
    datasetLabels = []
    # sepsisFiles = []
    for dir, path, files in os.walk(parameters['sepsisFilesPath']):
        for file in files:
            datasetFiles.append(dir + "/" + file)
            datasetLabels.append([1])

    # noSepsisFiles = []
    lenSepsisObjects = len(datasetFiles)
    for dir, path, files in os.walk(parameters['noSepsisFilesPath']):
        if len(datasetFiles) - lenSepsisObjects >= math.ceil(lenSepsisObjects * 1.5) :
            break
        for file in files:
            datasetFiles.append(dir + "/" + file)
            datasetLabels.append([0])

    print("========= Spliting data for testing")
    dataTrain, datasetFiles, labelsTrain, datasetLabels = train_test_split(datasetFiles, datasetLabels,
                                                                           stratify=datasetLabels, test_size=0.2)

    print("========= Saving dataset files array")
    with open(parameters['modelCheckpointPath']+parameters['datasetFilesFileName'], 'wb') as datasetFilesHandler:
        pickle.dump(datasetFiles, datasetFilesHandler, pickle.HIGHEST_PROTOCOL)

    with open(parameters['modelCheckpointPath']+parameters['datasetLabelsFileName'], 'wb') as datasetLabelsHandler:
        pickle.dump(datasetLabels, datasetLabelsHandler, pickle.HIGHEST_PROTOCOL)

if len(datasetFiles) == 0 or len(datasetLabels) == 0:
    raise ValueError("Dataset files is empty!")

#TODO : mudar aqui para ler o arquivo e filtrar os dados até hora da infecção - parameters['hoursBeforeInfectionPoe']

print("========= Loading sepsis3-df-no-exclusions.csv")
sepsis3_df = pd.read_csv('../sepsis3-df-no-exclusions.csv')
sepsis3_df = sepsis3_df[sepsis3_df['sepsis-3'] == 1]
print("========= Loading data from files")
data = []
for filePath, label in zip(datasetFiles, datasetLabels):
    patient_matrix = []
    csvObject = pd.read_csv(filePath)
    print(csvObject.keys())
    break

exit(1)

# TODO: FINISH THE DATA READING AND PREPROCESSING
labels = datasetLabels

#TODO: proper preprocess the data
print("========= Preprocessing data")
new_data = []
for d in data:
    new_data.append(regexp_tokenize(d.lower(), pattern='\w+|\$[\d\.]+|\S+'))

data = numpy.array(new_data)
labels = numpy.array(labels)


kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=15)
inputShape = (parameters['maxWords'], parameters['embeddingSize'])

i = 1
labelsForStratifiedKFold = []
for label in labels:
    labelsForStratifiedKFold.append(label[0])

config = None
if os.path.exists(parameters['modelConfigPath']):
    with open(parameters['modelConfigPath'], 'r') as configHandler:
        config = json.load(configHandler)

# ====================== Script that start training new models
with open(parameters['resultFilePath'], 'a+') as cvsFileHandler: # where the results for each fold are appended
    dictWriter = None
    for trainIndex, testIndex in kf.split(data, labelsForStratifiedKFold):
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
            # Training new fold
            print("========= Training word2vec")
            word2vecModel = gensim.models.Word2Vec(data[trainIndex], size = parameters['embeddingSize'], min_count=1,
                                                   window=parameters['word2vecWindow'],
                                                   iter=parameters['wordd2vecIter'], sg=1)
            dataTrainGenerator = data_transform('./data', data[trainIndex], labels[trainIndex], word2vecModel, parameters['embeddingSize'],
                                                1, maxWords=parameters['maxWords'])
            dataTestGenerator = data_transform('./data_test', data[testIndex], labels[testIndex], word2vecModel, parameters['embeddingSize'],
                                               1, maxWords=parameters['maxWords'])
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

with open(parameters['resultFilePath'], 'w') as cvsFileHandler:
    dictWriter = csv.DictWriter(cvsFileHandler, metrics_fold[0].keys())
    dictWriter.writeheader()
    for result in metrics_fold:
        dictWriter.writerow(result)
