import csv
import json
import os
import pickle
from collections import Counter
from pprint import PrettyPrinter

import pandas as pd
import numpy as np

import keras
from keras import backend as K
import tensorflow as tf
from keras.regularizers import l1_l2

from sklearn.model_selection._split import StratifiedKFold, train_test_split
from sklearn.utils import class_weight

from adapter import KerasAdapter
from multiclassification.parameters.classification_parameters import timeseries_training_parameters as parameters

import functions
from data_generators import LengthLongitudinalDataGenerator, LongitudinalDataGenerator
from functions import test_model, print_with_time
from keras_callbacks import Metrics
from model_creators import MultilayerKerasRecurrentNNCreator, MultilayerTemporalConvolutionalNNCreator
from normalization import Normalization, NormalizationValues

def focal_loss(y_true, y_pred):
    gamma = 2.0
    alpha = 0.25
    pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
    pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
    return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1))-K.sum((1-alpha) * K.pow( pt_0, gamma) * K.log(1. - pt_0))

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
DATETIME_PATTERN = "%Y-%m-%d %H:%M:%S"

problem = 'mortality'
training_base_directory = parameters['multiclassification_base_path'] + parameters['training_directory_path']
training_directory = training_base_directory + parameters[problem+"_directory"] \
                     + parameters['execution_saving_path']
checkpoint_directory = training_directory + parameters['training_checkpoint']
if not os.path.exists(checkpoint_directory):
    os.makedirs(checkpoint_directory)
with open(checkpoint_directory + parameters['execution_parameters_filename'], 'wb') as handler:
    pickle.dump(parameters, handler)

# Loading csv
print_with_time("Loading data")
dataset_path = parameters['multiclassification_base_path'] + parameters[problem+'_directory'] \
               + parameters[problem+'_dataset_csv']

data_csv = pd.read_csv(dataset_path)
data_csv = data_csv.sort_values(['episode'])
# Get the paths for the files
data = np.array(data_csv['structured_path'].tolist())
print_with_time("Class distribution")
print(data_csv['label'].value_counts())
classes = np.array(data_csv['label'].tolist())
data, X_val, classes, classes_evaluation = train_test_split(data, classes, stratify=classes,
                                                             test_size=.10)
print(pd.Series(classes).value_counts())
print_with_time("Computing class weights")
class_weights = class_weight.compute_class_weight('balanced',
                                                 np.unique(classes),
                                                 classes)
mapped_weights = dict()
for value in np.unique(classes):
    mapped_weights[value] = class_weights[value]
class_weights = mapped_weights


# Using a seed always will get the same data split even if the training stops
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=15)

print_with_time("Preparing normalization values")
normalization_value_counts_path = training_directory + parameters['normalization_value_counts_directory']
normalization_values = NormalizationValues(data, pickle_object_path=normalization_value_counts_path)
normalization_values.prepare()
# Get input shape
aux = pd.read_csv(data[0])
aux = functions.remove_columns_for_classification(aux)
inputShape = (None, len(aux.columns))
i = 0
# ====================== Script that start training new models
result_file_path = checkpoint_directory + parameters['result_filename']
with open(result_file_path, 'a+') as cvsFileHandler: # where the results for each fold are appended
    dictWriter = None
    for trainIndex, testIndex in kf.split(data, classes):
        trained_model_path = checkpoint_directory + parameters['trained_model_filename'].format(i)
        # if os.path.exists(trained_model_path):
        #     print("Pass fold {}".format(i))
        #     i += 1
        #     continue
        print_with_time("Fold {}".format(i))
        print_with_time("Getting values for normalization")
        # normalization_values = Normalization.get_normalization_values(data[trainIndex])
        fold_normalization_values_path = training_directory + parameters['fold_normalization_values_filename'].format(i)
        values = normalization_values.get_normalization_values(data[trainIndex],
                                                               saved_file_name=fold_normalization_values_path)
        fold_normalization_temporary_data_path = training_directory + parameters['fold_normalization_temporary_data_directory'].format(i)
        normalizer = Normalization(values, temporary_path=fold_normalization_temporary_data_path)
        print_with_time("Normalizing fold data")
        normalizer.normalize_files(data)
        normalized_data = np.array(normalizer.get_new_paths(data))
        print_with_time("Creating generators")
        # dataTrainGenerator = LongitudinalDataGenerator(normalized_data[trainIndex],
        #                                                classes[trainIndex], parameters['batchSize'])
        # dataTestGenerator = LongitudinalDataGenerator(normalized_data[testIndex],
        #                                               classes[testIndex], parameters['batchSize'])

        training_events_sizes_file_path = training_directory + parameters['training_events_sizes_filename'].format(i)
        training_events_sizes_labels_file_path = training_directory + parameters['training_events_sizes_labels_filename'].format(i)
        testing_events_sizes_file_path = training_directory + parameters['testing_events_sizes_filename'].format(i)
        testing_events_sizes_labels_file_path = training_directory + parameters['testing_events_sizes_labels_filename'].format(i)

        train_sizes, train_labels = functions.divide_by_events_lenght(normalized_data[trainIndex]
                                                                      , classes[trainIndex]
                                                                      , sizes_filename=training_events_sizes_file_path
                                                                      , classes_filename=training_events_sizes_labels_file_path)
        test_sizes, test_labels = functions.divide_by_events_lenght(normalized_data[testIndex], classes[testIndex]
                                                            , sizes_filename = testing_events_sizes_file_path
                                                            , classes_filename = testing_events_sizes_labels_file_path)

        dataTrainGenerator = LengthLongitudinalDataGenerator(train_sizes, train_labels, max_batch_size=parameters['batchSize'])
        dataTrainGenerator.create_batches()
        dataTestGenerator = LengthLongitudinalDataGenerator(test_sizes, test_labels, max_batch_size=parameters['batchSize'])
        dataTestGenerator.create_batches()
        if not os.path.exists(trained_model_path):
            if not parameters['tcn']:
                modelCreator = MultilayerKerasRecurrentNNCreator(inputShape, parameters['outputUnits'], parameters['numOutputNeurons'],
                                                                 loss=parameters['loss'], layersActivations=parameters['layersActivations'],
                                                                 networkActivation=parameters['networkActivation'],
                                                                 gru=parameters['gru'], use_dropout=parameters['useDropout'],
                                                                 dropout=parameters['dropout'], kernel_regularizer=None,
                                                                 metrics=[keras.metrics.binary_accuracy], optimizer=parameters['optimizer'])
            else:
                modelCreator = MultilayerTemporalConvolutionalNNCreator(inputShape, parameters['outputUnits'],
                                                                    parameters['numOutputNeurons'],
                                                                    loss=parameters['loss'],
                                                                    layersActivations=parameters['layersActivations'],
                                                                    networkActivation=parameters['networkActivation'],
                                                                    pooling=parameters['pooling'],
                                                                    kernel_sizes= parameters['kernel_sizes'],
                                                                    use_dropout=parameters['useDropout'],
                                                                    dilations=parameters['dilations'],
                                                                    nb_stacks=parameters['nb_stacks'],
                                                                    dropout=parameters['dropout'], kernel_regularizer=None,
                                                                    metrics=[keras.metrics.binary_accuracy],
                                                                    optimizer=parameters['optimizer'])
            kerasAdapter = modelCreator.create(model_summary_filename=checkpoint_directory+'model_summary.txt')
            epochs = parameters['trainingEpochs']
            metrics_callback = Metrics(dataTestGenerator)
            print_with_time("Training model")
            kerasAdapter.fit(dataTrainGenerator, epochs=epochs, callbacks=None, class_weights=None)
            kerasAdapter.save(trained_model_path)
        else:
            kerasAdapter = KerasAdapter.load_model(trained_model_path)
        print_with_time("Testing model")
        metrics = test_model(kerasAdapter, dataTestGenerator, i)
        if dictWriter is None:
            dictWriter = csv.DictWriter(cvsFileHandler, metrics.keys())
        if metrics['fold'] == 0:
            dictWriter.writeheader()
        dictWriter.writerow(metrics)
        i += 1

# Evaluating k-fold
results = pd.read_csv(result_file_path)
print(results.describe())
