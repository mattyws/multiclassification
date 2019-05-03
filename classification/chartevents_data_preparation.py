import json
import os
import pickle
from datetime import datetime, timedelta

import pandas as pd
from classification import chartevents_features
import math

import numpy
from sklearn.model_selection import train_test_split

from classification import chartevents_features
from data_generators import EmbeddingObjectSaver



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

dataset = pd.read_csv(parameters['datasetCsvFilePath'])
features = [int(key) for key in list(chartevents_features.FEATURES.keys())]
features.sort()

if not os.path.exists(parameters['dataPath']):
    os.mkdir(parameters['dataPath'])

classes = [1 if c == 'sepsis' else 0 for c in dataset['class'].tolist()]

print(" ======= Selecting random sample ======= ")
dataset, dataTest, classes, labelsTest = train_test_split(dataset['icustay_id'].tolist(), classes,
                                                                stratify=classes, test_size=0.8)

for icustayid, icu_class in zip(dataset, classes):
    if not os.path.exists(parameters['datasetFilesPath']+'{}.csv'.format(icustayid)):
        continue
    # Loading events
    events = pd.read_csv(parameters['datasetFilesPath']+'{}.csv'.format(icustayid))
    # Filtering
    events = events[events['itemid'].isin(features)]
    # Now add the events that doesn't appear with as empty row
    ids_in_events = events['itemid'].tolist()
    itemids_notin_events = [itemid if itemid not in ids_in_events else None for itemid in features]
    for id in itemids_notin_events:
        if id is not None:
            events = events.append({'itemid': id}, ignore_index=True)
    events['itemid'] = events['itemid'].astype(int)
    # The data representation is the features ordered by id
    events = events.set_index(['itemid']).sort_index()
    # Have to filter data based on the class
    if icu_class == 1:
        # If patient has sepsis, get a window of 8h
    print(events)
    break

exit()

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