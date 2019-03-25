import json
import os
import pickle
from datetime import datetime, timedelta

import pandas as pd
import chartevents_features
import math

import numpy
from sklearn.model_selection import train_test_split

from data_generators import Word2VecTextEmbeddingGenerator, EmbeddingObjectSaver


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

print("========= Loading data from files")
objectSaver = EmbeddingObjectSaver(parameters['allDataPath'])
data = []
time_before_suspicious_timedelta = timedelta(hours=parameters['hoursBeforeInfectionPoe'])
for filePath, label in zip(datasetFiles, datasetLabels):
    hadm_id = filePath.split('/')[-1].split('.')[0]
    # Get rows from sepsis-df that match with this hadm_id
    try:
        csvObject = pd.read_csv(filePath)
    except:
        print("Error in {}".format(filePath))
        exit(1)
    # Filter features
    csvObject = csvObject[csvObject['itemid'].isin(chartevents_features.FEATURES_ITEMS_LABELS.keys())]
    itemids = csvObject['itemid']
    for featureid in chartevents_features.FEATURES_ITEMS_LABELS.keys():
        if featureid not in itemids:
            csvObject = csvObject.append({'itemid' : featureid,
                                          'label': chartevents_features.FEATURES_ITEMS_LABELS[featureid]},
                                         ignore_index= True)
    # If its an sepsis patient, remove keys based on the suspicious of infection timestamp
    if label == 1:
        # Get supicious infection timestamp
        suspicous_series = csvObject[csvObject['itemid'] == -1].drop(columns=['itemid', 'label']).iloc[0].dropna()
        suspicious_timestamp = suspicous_series.keys()[0]
        suspiciousDatetime = datetime.strptime(suspicious_timestamp, DATETIME_PATTERN)
        # Drop unecessary columns and
        # Drop timestamp coluns that are not in the range of [admission_time, infection_time - hoursBeforeInfection]
        keysToDrop = []
        for key in csvObject.keys():
            try:
                keyDatetime = datetime.strptime(key, DATETIME_PATTERN)
                if keyDatetime > suspiciousDatetime - time_before_suspicious_timedelta:
                    keysToDrop.append(key)
            except:
                print(key)
        csvObject = csvObject.drop(columns=keysToDrop)
    csvObject = csvObject.drop(columns=['itemid', 'label'])
    dataMatrix = []
    for key in csvObject.keys():
        dataMatrix.append(csvObject[key])
    dataMatrix = numpy.transpose(dataMatrix)
    objectSaver.save((dataMatrix, label))