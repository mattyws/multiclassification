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
from tensorflow.keras.metrics import AUC

from adapter import KerasAdapter
from data_representation import ClinicalBertTextRepresentationTransform
from multiclassification.parameters.classification_parameters import timeseries_textual_training_parameters as parameters
from multiclassification.parameters.classification_parameters import model_tuner_parameters as tuner_parameters

import functions
from data_generators import LengthLongitudinalDataGenerator, LongitudinalDataGenerator
from functions import test_model, print_with_time
from keras_callbacks import Metrics
from model_creators import MultilayerKerasRecurrentNNCreator, MultilayerTemporalConvolutionalNNCreator, \
    KerasTunerModelCreator, MultilayerTemporalConvolutionalNNHyperModel
from normalization import Normalization, NormalizationValues
import kerastuner as kt

#TODO: Modify script to use text data

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


# Using a seed always will get the same data split even if the training stops
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=15)


text_transformer = ClinicalBertTextRepresentationTransform()
text_transformer.transform(data_csv, 'textual_path')



exit()

print_with_time("Preparing normalization values")
normalization_value_counts_path = training_directory + parameters['normalization_value_counts_directory']
normalization_values = NormalizationValues(data, pickle_object_path=normalization_value_counts_path)
normalization_values.prepare()
# Get input shape
aux = pd.read_csv(data[0])
aux = functions.remove_columns_for_classification(aux)
inputShape = (None, len(aux.columns))

if parameters['model_tunning']:
    training_samples_path = training_directory + parameters['training_samples_filename']
    training_classes_path = training_directory + parameters['training_classes_filename']
    optimization_samples_path = training_directory + parameters['optimization_samples_filename']
    optimization_classes_path = training_directory + parameters['optimization_classes_filename']

    if not os.path.exists(training_samples_path):
        data, data_opt, classes, classes_opt = train_test_split(data, classes, stratify=classes,
                                                                test_size=parameters['optimization_split_rate'])
        with open(training_samples_path, 'wb') as f:
            pickle.dump(data, f)
        with open(training_classes_path, 'wb') as f:
            pickle.dump(classes, f)
        with open(optimization_samples_path, 'wb') as f:
            pickle.dump(data_opt, f)
        with open(optimization_classes_path, 'wb') as f:
            pickle.dump(classes_opt, f)
    else:
        with open(training_samples_path, 'rb') as f:
            data = pickle.load(f)
        with open(training_classes_path, 'rb') as f:
            classes = pickle.load(f)
        with open(optimization_samples_path, 'rb') as f:
            data_opt = pickle.load(f)
        with open(optimization_classes_path, 'rb') as f:
            classes_opt = pickle.load(f)
    opt_normalization_values_path = training_directory + parameters['optimization_normalization_values_filename']
    values = normalization_values.get_normalization_values(data_opt,
                                                           saved_file_name=opt_normalization_values_path)
    opt_normalization_temporary_data_path = training_directory + parameters[
        'optimization_normalization_temporary_data_directory']
    normalizer = Normalization(values, temporary_path=opt_normalization_temporary_data_path)
    print_with_time("Normalizing optimization data")
    normalizer.normalize_files(data_opt)
    normalized_data = np.array(normalizer.get_new_paths(data_opt))
    print_with_time("Creating optimization generators")
    training_events_sizes_file_path = training_directory + parameters['training_events_sizes_filename'].format('opt')
    training_events_sizes_labels_file_path = training_directory + parameters[
        'training_events_sizes_labels_filename'].format('opt')
    testing_events_sizes_file_path = training_directory + parameters['testing_events_sizes_filename'].format('opt')
    testing_events_sizes_labels_file_path = training_directory + parameters[
        'testing_events_sizes_labels_filename'].format('opt')

    train_sizes, train_labels = functions.divide_by_events_lenght(normalized_data
                                                                  , classes_opt
                                                                  , sizes_filename=training_events_sizes_file_path
                                                                  ,
                                                                  classes_filename=training_events_sizes_labels_file_path)
    dataTrainGenerator = LengthLongitudinalDataGenerator(train_sizes, train_labels,
                                                         max_batch_size=parameters['batchSize'])
    dataTrainGenerator.create_batches()

    model_builder = MultilayerTemporalConvolutionalNNHyperModel(inputShape, parameters['numOutputNeurons'],
                                                                [AUC()], tuner_parameters)
    tunning_directory = checkpoint_directory + parameters['tunning_directory']
    tuner = kt.Hyperband(model_builder,
                         objective=kt.Objective('auc', direction="max"),
                         max_epochs=10,
                         directory=tunning_directory,
                         project_name='timeseries',
                         factor=3)
    tuner.search(dataTrainGenerator, epochs=10)
    modelCreator = KerasTunerModelCreator(tuner)
else:
    if not parameters['tcn']:
        modelCreator = MultilayerKerasRecurrentNNCreator(inputShape, parameters['outputUnits'],
                                                         parameters['numOutputNeurons'],
                                                         loss=parameters['loss'],
                                                         layersActivations=parameters['layersActivations'],
                                                         networkActivation=parameters['networkActivation'],
                                                         gru=parameters['gru'], use_dropout=parameters['useDropout'],
                                                         dropout=parameters['dropout'], kernel_regularizer=None,
                                                         metrics=[keras.metrics.binary_accuracy],
                                                         optimizer=parameters['optimizer'])
    else:
        modelCreator = MultilayerTemporalConvolutionalNNCreator(inputShape, parameters['outputUnits'],
                                                                parameters['numOutputNeurons'],
                                                                loss=parameters['loss'],
                                                                layersActivations=parameters['layersActivations'],
                                                                networkActivation=parameters['networkActivation'],
                                                                pooling=parameters['pooling'],
                                                                kernel_sizes=parameters['kernel_sizes'],
                                                                use_dropout=parameters['useDropout'],
                                                                dilations=parameters['dilations'],
                                                                nb_stacks=parameters['nb_stacks'],
                                                                dropout=parameters['dropout'], kernel_regularizer=None,
                                                                metrics=[keras.metrics.binary_accuracy],
                                                                optimizer=parameters['optimizer'])

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
