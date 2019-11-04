import csv
import json
import os

import pandas as pd
import numpy as np

import keras
from keras import backend as K
import tensorflow as tf
from keras.regularizers import l1_l2

from sklearn.model_selection._split import StratifiedKFold

import functions
from data_generators import LengthLongitudinalDataGenerator, LongitudinalDataGenerator
from functions import test_model, print_with_time
from keras_callbacks import Metrics
from model_creators import MultilayerKerasRecurrentNNCreator
from normalization import Normalization, NormalizationValues

def focal_loss(y_true, y_pred):
    gamma = 2.0
    alpha = 0.25
    pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
    pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
    return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1))-K.sum((1-alpha) * K.pow( pt_0, gamma) * K.log(1. - pt_0))

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
        # dataTrainGenerator = LongitudinalDataGenerator(normalized_data[trainIndex],
        #                                                classes[trainIndex], parameters['batchSize'],
        #                                                saved_batch_dir='training_batches_fold_{}'.format(i))
        # print(dataTrainGenerator[0][1])
        # print(dataTrainGenerator[0][0].shape)
        # print(dataTrainGenerator[0][1].shape)
        # exit()
        # dataTestGenerator = LongitudinalDataGenerator(normalized_data[testIndex],
        #                                               classes[testIndex], parameters['batchSize'],
        #                                               saved_batch_dir='testing_batches_fold_{}'.format(i))
        train_sizes, train_labels = functions.divide_by_events_lenght(normalized_data[trainIndex]
                                                                      , classes[trainIndex]
                                                                      , sizes_filename=parameters['training_events_sizes_file'].format(i)
                                                                      , classes_filename=parameters['training_events_sizes_labels_file'].format(i))
        test_sizes, test_labels = functions.divide_by_events_lenght(normalized_data[testIndex], classes[testIndex]
                                                            , sizes_filename = parameters['testing_events_sizes_file'].format(i)
                                                            , classes_filename = parameters['testing_events_sizes_labels_file'].format(i))


        # new_sizes = dict()
        # new_labels = dict()
        # i = 0
        # for key in train_sizes.keys():
        #     new_sizes[key] = train_sizes[key]
        #     new_labels[key] = train_labels[key]
        #     if i == 4:
        #         break
        #     i += 1
        # train_sizes = new_sizes
        # train_labels = new_labels
        # new_sizes = dict()
        # new_labels = dict()
        # i = 0
        # for key in test_sizes.keys():
        #     new_sizes[key] = test_sizes[key]
        #     new_labels[key] = test_labels[key]
        #     if i == 4:
        #         break
        #     i += 1
        # test_sizes = new_sizes
        # test_labels = new_labels


        dataTrainGenerator = LengthLongitudinalDataGenerator(train_sizes, train_labels, max_batch_size=parameters['batchSize'])
        dataTrainGenerator.create_batches()
        dataTestGenerator = LengthLongitudinalDataGenerator(test_sizes, test_labels, max_batch_size=parameters['batchSize'])
        dataTestGenerator.create_batches()
        # print(dataTrainGenerator[0][1])
        # print(dataTrainGenerator[0][0].shape)
        # print(dataTrainGenerator[0][1].shape)
        # for i in range(0, len(dataTrainGenerator)):
        #     print(len(dataTrainGenerator[i][0]))
        #     print(dataTrainGenerator[i][0])
        #     print(len(dataTrainGenerator[i][1]))
        #     print(dataTrainGenerator[i][1])
        #     input()


        modelCreator = MultilayerKerasRecurrentNNCreator(inputShape, parameters['outputUnits'], parameters['numOutputNeurons'],
                                                         loss=focal_loss, layersActivations=parameters['layersActivations'],
                                                         gru=parameters['gru'], use_dropout=parameters['useDropout'],
                                                         dropout=parameters['dropout'], kernel_regularizer=l1_l2(l1=0.001, l2=0.01),
                                                         metrics=[keras.metrics.binary_accuracy], optimizer='nadam')
        kerasAdapter = modelCreator.create(model_summary_filename=parameters['modelCheckpointPath']+'model_summary')
        epochs = parameters['trainingEpochs']
        metrics_callback = Metrics(dataTestGenerator)
        print_with_time("Training model")
        kerasAdapter.fit(dataTrainGenerator, epochs=epochs, callbacks=None)
        print_with_time("Testing model")
        metrics = test_model(kerasAdapter, dataTestGenerator, i)
        if dictWriter is None:
            dictWriter = csv.DictWriter(cvsFileHandler, metrics.keys())
        if metrics['fold'] == 0:
            dictWriter.writeheader()
        dictWriter.writerow(metrics)
        i += 1
