import csv
import json
import os
import pickle
import pprint
from collections import Counter

import pandas as pd
import numpy as np

import keras
import sys
from sklearn.metrics import f1_score

from keras.callbacks import ModelCheckpoint
from sklearn.metrics.classification import precision_score, recall_score, accuracy_score, classification_report, \
    cohen_kappa_score, confusion_matrix
from sklearn.metrics.ranking import roc_auc_score
from sklearn.model_selection._split import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight

import functions
from data_generators import LongitudinalDataGenerator, LengthLongitudinalDataGenerator
from keras_callbacks import ResumeTrainingCallback
from model_creators import MultilayerKerasRecurrentNNCreator
from metrics import f1, precision, recall
from normalization import Normalization, NormalizationValues

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
DATETIME_PATTERN = "%Y-%m-%d %H:%M:%S"

def test_model(kerasAdapter, dataTestGenerator, testClasses, fold, parameters):
    result = kerasAdapter.predict(dataTestGenerator, batch_size=parameters['batchSize'])
    # testClasses = classes[testIndex]
    metrics = dict()
    metrics['fscore'] = f1_score(testClasses, result, average='weighted')
    metrics['precision'] = precision_score(testClasses, result, average='weighted')
    metrics['recall'] = recall_score(testClasses, result, average='weighted')
    metrics['auc'] = roc_auc_score(testClasses, result, average='weighted')

    metrics['fscore_b'] = f1_score(testClasses, result)
    metrics['precision_b'] = precision_score(testClasses, result)
    metrics['recall_b'] = recall_score(testClasses, result)
    metrics['auc_b'] = roc_auc_score(testClasses, result)

    metrics['kappa'] = cohen_kappa_score(testClasses, result)
    metrics['accuracy'] = accuracy_score(testClasses, result)
    tn, fp, fn, metrics['tp_rate'] = confusion_matrix(testClasses, result).ravel()
    print(classification_report(testClasses, result))
    metrics["fold"] = fold
    return metrics



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
print("========= Loading data")
data_csv = pd.read_csv(parameters['datasetCsvFilePath'])
data_csv = data_csv.sort_values(['icustay_id'])
# Get the values in data_csv that have events saved
data = np.array([itemid for itemid in list(data_csv['icustay_id'])
                 if os.path.exists(parameters['dataPath'] + '{}.csv'.format(itemid))])
data_csv = data_csv[data_csv['icustay_id'].isin(data)]
data = np.array([parameters['dataPath'] + '{}.csv'.format(itemid) for itemid in data])
print("========= Transforming classes")
classes = np.array([1 if c == 'sepsis' else 0 for c in list(data_csv['class'])])
classes_for_stratified = np.array([1 if c == 'sepsis' else 0 for c in list(data_csv['class'])])
# Using a seed always will get the same data split even if the training stops
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=15)

print("========= Preparing normalization values")
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
        if config is not None and config['epoch'] == parameters['trainingEpochs']:
            print("Training reach the max of epochs for fold {}, testing last generated model".format(i))
            print("========= Loading generators")
            with open(parameters['trainingGeneratorPath'], 'rb') as trainingGeneratorHandler:
                dataTrainGenerator = pickle.load(trainingGeneratorHandler)

            with open(parameters['testingGeneratorPath'], 'rb') as testingGeneratorHandler:
                dataTestGenerator = pickle.load(testingGeneratorHandler)

            kerasAdapter = MultilayerKerasRecurrentNNCreator.create_from_path(config['filepath'],
                                                                              custom_objects={'f1': f1,
                                                                                              'precision': precision,
                                                                                              'recall': recall})
            metrics = test_model(kerasAdapter, dataTestGenerator, classes[testIndex], i, parameters)
            if dictWriter is None:
                dictWriter = csv.DictWriter(cvsFileHandler, metrics.keys())
            if metrics['fold'] == 0:
                dictWriter.writeheader()
            dictWriter.writerow(metrics)
            i += 1
            continue
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
            configSaver = ResumeTrainingCallback(parameters['modelConfigPath'],
                                                 parameters['modelCheckpointPath'] + 'fold_' + str(i), i, alreadyTrainedEpochs=config['epoch'])
        else:
            print("===== Getting values for normalization =====")
            # normalization_values = Normalization.get_normalization_values(data[trainIndex])
            values = normalization_values.get_normalization_values(data[trainIndex],
                                                                   saved_file_name=parameters['normalization_data_path'].format(i))
            normalizer = Normalization(values, temporary_path=parameters['temporary_data_path'].format(i))
            print("===== Normalizing fold data =====")
            normalizer.normalize_files(data)
            normalized_data = np.array(normalizer.get_new_paths(data))
            print("### Getting sizes ###")
            train_sizes, train_labels = functions.divide_by_events_lenght(normalized_data[trainIndex]
                                                                          , classes[trainIndex]
                                                                          , sizes_filename=parameters['training_events_sizes_file']
                                                                          , classes_filename=parameters['training_events_sizes_labels_file'])
            test_sizes, test_labels = functions.divide_by_events_lenght(normalized_data[testIndex], classes[testIndex]
                                                                , sizes_filename = parameters['testing_events_sizes_file']
                                                                , classes_filename = parameters['testing_events_sizes_labels_file'])
            dataTrainGenerator = LengthLongitudinalDataGenerator(train_sizes, train_labels)
            dataTrainGenerator.create_batches()
            dataTestGenerator = LengthLongitudinalDataGenerator(test_sizes, test_labels)
            dataTestGenerator.create_batches()
            # dataTrainGenerator = LongitudinalDataGenerator(normalized_data[trainIndex],
            #                                                classes[trainIndex], parameters['batchSize'],
            #                                                saved_batch_dir='training_batches_fold_{}'.format(i))
            # dataTestGenerator = LongitudinalDataGenerator(normalized_data[testIndex],
            #                                               classes[testIndex], parameters['batchSize'],
            #                                               saved_batch_dir='testing_batches_fold_{}'.format(i))
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
            kerasAdapter = modelCreator.create(model_summary_filename=parameters['modelCheckpointPath']+'model_summary')
            epochs = parameters['trainingEpochs']
            configSaver = ResumeTrainingCallback(parameters['modelConfigPath'],
                                                 parameters['modelCheckpointPath'] + 'fold_' + str(i), i)

        modelCheckpoint = ModelCheckpoint(parameters['modelCheckpointPath']+'fold_'+str(i))
        kerasAdapter.fit(dataTrainGenerator, epochs=epochs, batch_size=len(dataTrainGenerator),
                         validationSteps=len(dataTestGenerator),
                         callbacks=[modelCheckpoint, configSaver])
        metrics = test_model(kerasAdapter, dataTestGenerator, classes[testIndex], i, parameters)
        if dictWriter is None:
            dictWriter = csv.DictWriter(cvsFileHandler, metrics.keys())
        if metrics['fold'] == 0:
            dictWriter.writeheader()
        dictWriter.writerow(metrics)
        i += 1
