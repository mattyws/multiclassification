import csv
import json
import os

import pandas as pd
import numpy as np

import keras

from sklearn.model_selection._split import StratifiedKFold

import functions
from data_generators import LengthLongitudinalDataGenerator
from functions import test_model, print_with_time
from keras_callbacks import Metrics
from model_creators import MultilayerKerasRecurrentNNCreator
from normalization import Normalization, NormalizationValues

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
DATETIME_PATTERN = "%Y-%m-%d %H:%M:%S"

parametersFilePath = "./classification_parameters.json"

#Loading parameters file
print("========= Loading Parameters")
parameters = None
with open(parametersFilePath, 'r') as parametersFileHandler:
    parameters = json.load(parametersFileHandler)
if parameters is None:
    exit(1)

if not os.path.exists(parameters['modelCheckpointPath']):
    os.mkdir(parameters['modelCheckpointPath'])

config = None
if os.path.exists(parameters['modelConfigPath']):
    with open(parameters['modelConfigPath'], 'r') as configHandler:
        config = json.load(configHandler)

# Loading csv
print_with_time("Loading data")
data_csv = pd.read_csv(parameters['datasetCsvFilePath'])
data_csv = data_csv.sort_values(['icustay_id'])
# Get the values in data_csv that have events saved
data = np.array([itemid for itemid in list(data_csv['icustay_id'])
                 if os.path.exists(parameters['dataPath'] + '{}.csv'.format(itemid))])
data_csv = data_csv[data_csv['icustay_id'].isin(data)]
data = np.array([parameters['dataPath'] + '{}.csv'.format(itemid) for itemid in data])
print_with_time("Transforming classes")
classes = np.array([1 if c == 'sepsis' else 0 for c in list(data_csv['class'])])
classes_for_stratified = np.array([1 if c == 'sepsis' else 0 for c in list(data_csv['class'])])
# Using a seed always will get the same data split even if the training stops
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=15)

print_with_time("Preparing normalization values")
normalization_values = NormalizationValues(data, pickle_object_path=parameters['normalization_value_counts_path'])
normalization_values.prepare()
# Get input shape
aux = pd.read_csv(data[0])
if 'Unnamed: 0' in aux.columns:
    aux = aux.drop(columns=['Unnamed: 0'])
if 'chartevents_Unnamed: 0' in aux.columns:
    aux = aux.drop(columns=['chartevents_Unnamed: 0'])
if 'labevents_Unnamed: 0' in aux.columns:
    aux = aux.drop(columns=['labevents_Unnamed: 0'])
if 'starttime' in aux.columns:
    aux = aux.drop(columns=['starttime'])
if 'endtime' in aux.columns:
    aux = aux.drop(columns=['endtime'])
inputShape = (None, len(aux.columns))
i = 0
# ====================== Script that start training new models
with open(parameters['resultFilePath'], 'a+') as cvsFileHandler: # where the results for each fold are appended
    dictWriter = None
    for trainIndex, testIndex in kf.split(data, classes):
        if config is not None and config['fold'] > i:
            print("Pass fold {}".format(i))
            i += 1
            continue
        print_with_time("Fold {}".format(i))
        print_with_time("Getting values for normalization")
        # normalization_values = Normalization.get_normalization_values(data[trainIndex])
        values = normalization_values.get_normalization_values(data[trainIndex],
                                                               saved_file_name=parameters['normalization_data_path'].format(i))
        normalizer = Normalization(values, temporary_path=parameters['temporary_data_path'].format(i))
        print_with_time("Normalizing fold data")
        normalizer.normalize_files(data)
        normalized_data = np.array(normalizer.get_new_paths(data))
        print_with_time("Creating generators")
        train_sizes, train_labels = functions.divide_by_events_lenght(normalized_data[trainIndex]
                                                                      , classes[trainIndex]
                                                                      , sizes_filename=parameters['training_events_sizes_file'].format(i)
                                                                      , classes_filename=parameters['training_events_sizes_labels_file'].format(i))
        test_sizes, test_labels = functions.divide_by_events_lenght(normalized_data[testIndex], classes[testIndex]
                                                            , sizes_filename = parameters['testing_events_sizes_file'].format(i)
                                                            , classes_filename = parameters['testing_events_sizes_labels_file'].format(i))
        dataTrainGenerator = LengthLongitudinalDataGenerator(train_sizes, train_labels, max_batch_size=parameters['batchSize'])
        dataTrainGenerator.create_batches()
        dataTestGenerator = LengthLongitudinalDataGenerator(test_sizes, test_labels, max_batch_size=parameters['batchSize'])
        dataTestGenerator.create_batches()
        for i in range(0, len(dataTrainGenerator)):
            print(len(dataTrainGenerator[i][0]))
            print(len(dataTrainGenerator[i][1]))
            input()
            for j in len(dataTrainGenerator[i][0]):
                print(dataTrainGenerator[i][0][j])
                print(dataTrainGenerator[i][1][j])
                print(len(data))
                input()
        # dataTrainGenerator = LongitudinalDataGenerator(normalized_data[trainIndex],
        #                                                classes[trainIndex], parameters['batchSize'],
        #                                                saved_batch_dir='training_batches_fold_{}'.format(i))
        # dataTestGenerator = LongitudinalDataGenerator(normalized_data[testIndex],
        #                                               classes[testIndex], parameters['batchSize'],
        #                                               saved_batch_dir='testing_batches_fold_{}'.format(i))

        modelCreator = MultilayerKerasRecurrentNNCreator(inputShape, parameters['outputUnits'], parameters['numOutputNeurons'],
                                                         loss=parameters['loss'], layersActivations=parameters['layersActivations'],
                                                         gru=parameters['gru'], use_dropout=parameters['useDropout'],
                                                         dropout=parameters['dropout'],
                                                         metrics=[keras.metrics.binary_accuracy])
        kerasAdapter = modelCreator.create(model_summary_filename=parameters['modelCheckpointPath']+'model_summary')
        epochs = parameters['trainingEpochs']
        metrics_callback = Metrics(dataTestGenerator)
        kerasAdapter.fit(dataTrainGenerator, epochs=epochs, callbacks=[metrics_callback])
        metrics = test_model(kerasAdapter, dataTestGenerator, i)
        if dictWriter is None:
            dictWriter = csv.DictWriter(cvsFileHandler, metrics.keys())
        if metrics['fold'] == 0:
            dictWriter.writeheader()
        dictWriter.writerow(metrics)
        i += 1
