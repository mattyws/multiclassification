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
from multiclassification.parameters.classification_parameters import timeseries_training_parameters as parameters
from multiclassification.parameters.classification_parameters import model_tuner_parameters as tuner_parameters

from resources import functions
from resources.data_generators import LengthLongitudinalDataGenerator, LongitudinalDataGenerator
from resources.functions import test_model, print_with_time
from resources.keras_callbacks import Metrics
from resources.model_creators import MultilayerKerasRecurrentNNCreator, MultilayerTemporalConvolutionalNNCreator, \
    KerasTunerModelCreator, MultilayerTemporalConvolutionalNNHyperModel
from resources.normalization import Normalization, NormalizationValues
import kerastuner as kt

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
episodes = data_csv['episode'].tolist()
print_with_time("Class distribution")
classes = np.array(data_csv['label'].tolist())
print_with_time("Computing class weights")
class_weights = class_weight.compute_class_weight('balanced',
                                                 np.unique(classes),
                                                 classes)
mapped_weights = dict()
for value in np.unique(classes):
    mapped_weights[value] = class_weights[value]
class_weights = mapped_weights
class_weights = None


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

training_samples_path = training_directory + parameters['training_samples_filename']
training_classes_path = training_directory + parameters['training_classes_filename']
testing_samples_path = training_directory + parameters['testing_samples_filename']
testing_classes_path = training_directory + parameters['testing_classes_filename']

optimization_samples_path = training_directory + parameters['optimization_samples_filename']
optimization_classes_path = training_directory + parameters['optimization_classes_filename']

if not os.path.exists(training_samples_path):
    len_dataset = len(data_csv)
    len_evaluation_dataset = int(len_dataset * parameters['train_test_split_rate'])
    len_optimization_dataset = int(len_dataset * parameters['optimization_split_rate'])
    data, data_val, classes, classes_evaluation = train_test_split(data_csv, classes, stratify=classes,
                                                             test_size=len_evaluation_dataset)

    #############################################
    ### Balancing instances on trainning data ###
    #############################################

    len_positive = len(data[data['label'] == 1])
    subsample = data[data['label'] == 0].sample(len_positive)
    data = subsample.append(data[data['label'] == 1])
    classes = data['label'].tolist()

    data, data_opt, classes, classes_opt = train_test_split(data, classes, stratify=classes,
                                                      test_size=len_optimization_dataset)
    with open(training_samples_path, 'wb') as f:
        pickle.dump(data, f)
    with open(training_classes_path, 'wb') as f:
        pickle.dump(classes, f)
    with open(testing_samples_path, 'wb') as f:
        pickle.dump(data_val, f)
    with open(testing_classes_path, 'wb') as f:
        pickle.dump(classes_evaluation, f)
    with open(optimization_samples_path, 'wb') as f :
        pickle.dump(data_opt, f)
    with open(optimization_classes_path, 'wb') as f:
        pickle.dump(classes_opt, f)
else:
    with open(training_samples_path, 'rb') as f:
        data = pickle.load(f)
    with open(training_classes_path, 'rb') as f:
        classes = pickle.load(f)
    with open(testing_samples_path, 'rb') as f:
        data_val = pickle.load(f)
    with open(testing_classes_path, 'rb') as f:
        classes_evaluation = pickle.load(f)
    with open(optimization_samples_path, 'rb') as f :
        data_opt = pickle.load(f)
    with open(optimization_classes_path, 'rb') as f:
        classes_opt = pickle.load(f)

optimization_df = data_opt #data_csv[data_csv['episode'].isin(data_opt)]
classes_opt = np.asarray(optimization_df['label'].tolist())
optimization_sdata = np.asarray(optimization_df['structured_path'].tolist())
opt_normalization_values_path = training_directory + parameters['optimization_normalization_values_filename']
values = normalization_values.get_normalization_values(optimization_sdata,
                                                       saved_file_name=opt_normalization_values_path)
opt_normalization_temporary_data_path = training_directory + parameters[
    'optimization_normalization_temporary_data_directory']
normalizer = Normalization(values, temporary_path=opt_normalization_temporary_data_path)
print_with_time("Normalizing optimization data")
normalizer.normalize_files(optimization_sdata)
normalized_data = np.array(normalizer.get_new_paths(optimization_sdata))

print_with_time("Creating optimization generators")
training_events_sizes_file_path = training_directory + parameters['training_events_sizes_filename'].format('opt')
training_events_sizes_labels_file_path = training_directory + parameters[
    'training_events_sizes_labels_filename'].format('opt')

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
modelCreator = KerasTunerModelCreator(tuner, "LSTM_ml")


evaluation_data = np.asarray(data_val['structured_path'].tolist())
eval_normalization_values_path = training_directory + parameters['evaluation_normalization_values_filename']
eval_values = normalization_values.get_normalization_values(evaluation_data,
                                                       saved_file_name=eval_normalization_values_path)
eval_normalization_temporary_data_path = training_directory + parameters[
    'evaluation_normalization_temporary_data_directory']
eval_normalizer = Normalization(eval_values, temporary_path=eval_normalization_temporary_data_path)
print_with_time("Normalizing evaluation data")
eval_normalizer.normalize_files(evaluation_data)
eval_normalized_data = np.array(eval_normalizer.get_new_paths(evaluation_data))

evaluation_events_sizes_file_path = training_directory + parameters['evaluation_events_sizes_filename']
evaluation_events_sizes_labels_file_path = training_directory + parameters['evaluation_events_sizes_labels_filename']
evaluation_sizes, evaluation_labels = functions.divide_by_events_lenght(eval_normalized_data
                                                                      , data_val['label'].tolist()
                                                                      , sizes_filename=training_events_sizes_file_path
                                                                      , classes_filename=training_events_sizes_labels_file_path)
evaluationGenerator = LengthLongitudinalDataGenerator(evaluation_sizes, evaluation_labels, max_batch_size=parameters['batchSize'])
evaluationGenerator.create_batches()
classes = np.asarray(data['label'].tolist())
data = np.asarray(data['structured_path'].tolist())

fold = 0
# ====================== Script that start training new models
result_file_path = checkpoint_directory + parameters['result_filename']
eval_file_path = checkpoint_directory + "eval_results.csv"
with open(result_file_path, 'a+') as cvsFileHandler, open(eval_file_path, 'a+') as evalFileHandler: # where the results for each fold are appended
    dictWriter = None
    eval_results = None
    for trainIndex, testIndex in kf.split(data, classes):
        trained_model_path = checkpoint_directory + parameters['trained_model_filename'].format(fold)
        if os.path.exists(trained_model_path):
            print("Pass fold {}".format(fold))
            fold += 1
            continue
        print_with_time("Fold {}".format(fold))
        print_with_time("Getting values for normalization")
        # normalization_values = Normalization.get_normalization_values(data[trainIndex])
        fold_normalization_values_path = training_directory + parameters['fold_normalization_values_filename'].format(fold)
        values = normalization_values.get_normalization_values(data[trainIndex],
                                                               saved_file_name=fold_normalization_values_path)
        fold_normalization_temporary_data_path = training_directory + parameters['fold_normalization_temporary_data_directory'].format(fold)
        normalizer = Normalization(values, temporary_path=fold_normalization_temporary_data_path)
        print_with_time("Normalizing fold data")
        normalizer.normalize_files(data)
        normalized_data = np.array(normalizer.get_new_paths(data))
        print_with_time("Creating generators")
        # dataTrainGenerator = LongitudinalDataGenerator(normalized_data[trainIndex],
        #                                                classes[trainIndex], parameters['batchSize'])
        # dataTestGenerator = LongitudinalDataGenerator(normalized_data[testIndex],
        #                                               classes[testIndex], parameters['batchSize'])

        training_events_sizes_file_path = training_directory + parameters['training_events_sizes_filename'].format(fold)
        training_events_sizes_labels_file_path = training_directory + parameters['training_events_sizes_labels_filename'].format(fold)
        testing_events_sizes_file_path = training_directory + parameters['testing_events_sizes_filename'].format(fold)
        testing_events_sizes_labels_file_path = training_directory + parameters['testing_events_sizes_labels_filename'].format(fold)

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
        metrics = test_model(kerasAdapter, dataTestGenerator, fold)
        keys = list(metrics.keys())
        keys.sort()
        if dictWriter is None:
            dictWriter = csv.DictWriter(cvsFileHandler, keys)
        if fold == 0:
            dictWriter.writeheader()
        dictWriter.writerow(metrics)


        metrics = test_model(kerasAdapter, evaluationGenerator, fold)
        keys = list(metrics.keys())
        keys.sort()
        if eval_results is None:
            eval_results = csv.DictWriter(evalFileHandler, keys)
        if fold == 0:
            eval_results.writeheader()
        eval_results.writerow(metrics)
        fold += 1

# Evaluating k-fold
results = pd.read_csv(result_file_path)
print(results.describe())
