import csv
import json
import os
from collections import Counter
from pprint import PrettyPrinter

import pandas as pd
import numpy as np

import keras
from keras import backend as K
import tensorflow as tf
from keras.regularizers import l1_l2

from sklearn.model_selection._split import StratifiedKFold

import functions
from adapter import KerasGeneratorAdapter
from data_generators import LengthLongitudinalDataGenerator, LongitudinalDataGenerator, MetaLearnerDataGenerator
from data_representation import EnsembleMetaLearnerDataCreator
from ensemble_training import TrainEnsembleAdaBoosting, TrainEnsembleBagging
from functions import test_model, print_with_time
from keras_callbacks import Metrics
from model_creators import MultilayerKerasRecurrentNNCreator, EnsembleModelCreator
from normalization import Normalization, NormalizationValues

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
DATETIME_PATTERN = "%Y-%m-%d %H:%M:%S"

parametersFilePath = "./classification_ensemble_parameters.json"

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
fold = 0
# ====================== Script that start training new models
with open(parameters['resultFilePath'], 'a+') as cvsFileHandler, \
        open(parameters['level_zero_result_file_path']) as level_zero_csv_file_handler: # where the results for each fold are appended
    dictWriter = None
    level_zero_dict_writer = None
    for trainIndex, testIndex in kf.split(data, classes):
        if config is not None and config['fold'] > fold:
            print("Pass fold {}".format(fold))
            fold += 1
            continue
        print_with_time("Fold {}".format(fold))
        print_with_time("Getting values for normalization")
        values = normalization_values.get_normalization_values(data[trainIndex],
                                                               saved_file_name=parameters['normalization_data_path'].format(fold))
        normalizer = Normalization(values, temporary_path=parameters['temporary_data_path'].format(fold))
        print_with_time("Normalizing fold data")
        normalizer.normalize_files(data)
        normalized_data = np.array(normalizer.get_new_paths(data))

        print_with_time("Generating level 0 models")

        #### START ADABOOSTING ####
        # ensemble = TrainEnsembleAdaBoosting()
        # ensemble.fit()
        ### END ADABOOSTING ####

        #### START BAGGING ####
        ensemble = TrainEnsembleBagging(normalized_data[trainIndex], classes[trainIndex])
        ensemble.fit(epochs=parameters['level_0_epochs'])
        ### END ADABOOSTING ####

        ### START CLUSTERING ENSEMBLE ###
        ### END CLUSTERING ENSEMBLE ###
        print_with_time("Testing level 0 models")
        level_zero_models = ensemble.get_classifiers()
        #TODO: test classifiers on testing fold
        test_sizes, test_labels = functions.divide_by_events_lenght(normalized_data[testIndex], classes[testIndex])
        data_test_generator = LengthLongitudinalDataGenerator(test_sizes, test_labels,
                                                            max_batch_size=parameters['batchSize'])
        data_test_generator.create_batches()
        for level_zero_model in level_zero_models:
            adpter = KerasGeneratorAdapter.load_model(level_zero_model)

            metrics = test_model(adpter, data_test_generator, fold)
            if level_zero_dict_writer is None:
                level_zero_dict_writer = csv.DictWriter(level_zero_csv_file_handler, metrics.keys())
            if fold == 0 :
                level_zero_dict_writer.writeheader()
            level_zero_dict_writer.writerow(metrics)

        print_with_time("Creating meta model data")

        meta_data_creator = EnsembleMetaLearnerDataCreator(level_zero_models)
        meta_data_creator.create_meta_learner_data(normalized_data, parameters['meta_representation_path'])

        meta_data = meta_data_creator.get_new_paths(normalized_data)


        print_with_time("Creating meta data generators")

        training_meta_data_generator = MetaLearnerDataGenerator(meta_data[trainIndex], classes[trainIndex],
                                                       batchSize=parameters['meta_learner_batch_size'])
        testing_meta_data_generator = MetaLearnerDataGenerator(meta_data[testIndex], classes[testIndex],
                                                                batchSize=parameters['meta_learner_batch_size'])


        modelCreator = EnsembleModelCreator(inputShape, parameters['outputUnits'], parameters['numOutputNeurons'],
                                                         loss=parameters['loss'], layers_activation=parameters['layersActivations'],
                                                         network_activation=parameters['networkActivation'],
                                                         use_dropout=parameters['useDropout'],
                                                         dropout=parameters['dropout'],
                                                         metrics=[keras.metrics.binary_accuracy], optimizer=parameters['optimizer'])
        with open(parameters['modelCheckpointPath']+"parameters.json", 'w') as handler:
            json.dump(parameters, handler)
        kerasAdapter = modelCreator.create()
        epochs = parameters['trainingEpochs']
        print_with_time("Training model")
        kerasAdapter.fit(training_meta_data_generator, epochs=epochs, callbacks=None)
        print_with_time("Testing model")
        metrics = test_model(kerasAdapter, testing_meta_data_generator, fold)
        if dictWriter is None:
            dictWriter = csv.DictWriter(cvsFileHandler, metrics.keys())
        if fold == 0:
            dictWriter.writeheader()
        dictWriter.writerow(metrics)
        kerasAdapter.save(parameters['modelCheckpointPath'] + 'trained_{}.model'.format(fold))
        fold += 1
