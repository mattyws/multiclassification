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
from multiclassification.parameters.classification_parameters import cuis_training_parameters as parameters
from multiclassification.parameters.classification_parameters import model_tuner_parameters as tuner_parameters

from resources import functions
from resources.cuis_filtering import FilterCUI
from resources.data_generators import LengthLongitudinalDataGenerator, LongitudinalDataGenerator
from resources.functions import test_model, print_with_time
from resources.keras_callbacks import Metrics
from resources.model_creators import MultilayerKerasRecurrentNNCreator, MultilayerTemporalConvolutionalNNCreator, \
    KerasTunerModelCreator, MultilayerTemporalConvolutionalNNHyperModel
from resources.normalization import Normalization, NormalizationValues
import kerastuner as kt

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


problem_dir_path = os.path.join(parameters['multiclassification_base_path'], parameters[problem+'_directory'])
dataset_path = os.path.join(problem_dir_path, parameters[problem+'_dataset_csv'])

if not os.path.exists(os.path.join(training_directory, "dataset.csv")):
    data_csv = pd.read_csv(dataset_path)
    data_csv = data_csv.loc[:,~data_csv.columns.str.match("Unnamed")]
    data_csv = data_csv.sort_values(['episode'])

    tf_idf_dataset_path = os.path.join(problem_dir_path, "tfidf_paths.csv")
    tf_idf_csv = pd.read_csv(tf_idf_dataset_path)
    tf_idf_csv = tf_idf_csv.loc[:,~tf_idf_csv.columns.str.match("Unnamed")]
    tf_idf_csv.loc[:, "path"] = tf_idf_csv["path"].apply(lambda x: os.path.join(problem_dir_path, x))

    # TODO: remover normalização
    # TODO: filtrar os CUIs usando o inverse document frequency (quanto maior, melhor), lembrar de deixar aberto para extrar tanto os maiores quanto os menores
    # TODO: a partir disso, só alterar os locais de onde estão tirando os caminhos para os arquivos, e desse modo, a solução é a mesma que as dos dados longitudinais
    print_with_time("Extracting CUIs using the IDF")
    cuis_idf = pd.read_csv(os.path.join(problem_dir_path, "cuis_idf.csv"))
    cuis_idf = cuis_idf.set_index(["Unnamed: 0"])
    cuis_idf = cuis_idf["idf"]
    filtered_cuis_files_path = os.path.join(training_directory, "filtered_cuis")
    if not os.path.exists(filtered_cuis_files_path):
        os.makedirs(filtered_cuis_files_path)
    filter = FilterCUI(tf_idf_csv, 150, 150, filtered_cuis_files_path)

    tf_idf_csv = filter.filter(cuis_idf)
    data_csv = pd.merge(tf_idf_csv, data_csv[["episode", "label"]], left_on="episode", right_on="episode")
    data_csv.to_csv(os.path.join(training_directory, "dataset.csv"))
else:
    data_csv = pd.read_csv(os.path.join(training_directory, "dataset.csv"))

# Get the paths for the files
data = np.array(data_csv['path'].tolist())
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
if not parameters['use_class_weight']:
    class_weights = None



# Get input shape
with open(data[0], 'rb') as pkl_file:
    aux = pickle.load(pkl_file)
inputShape = (None, len(aux[0]))

training_samples_path = training_directory + parameters['training_samples_filename']
training_classes_path = training_directory + parameters['training_classes_filename']
testing_samples_path = training_directory + parameters['testing_samples_filename']
testing_classes_path = training_directory + parameters['testing_classes_filename']

optimization_samples_path = training_directory + parameters['optimization_samples_filename']
optimization_classes_path = training_directory + parameters['optimization_classes_filename']

if not os.path.exists(training_samples_path):
    len_dataset = len(data_csv)
    len_optimization_dataset = int(len_dataset * parameters['optimization_split_rate'])
    data, data_opt, classes, classes_opt = train_test_split(episodes, classes, stratify=classes,
                                                            test_size=len_optimization_dataset)

    if parameters['balance_training_data']:
        #############################################
        ### Balancing instances on trainning data ###
        #############################################
        aux = data_csv[data_csv['episode'].isin(data)]
        len_positive = len(aux[aux['label'] == 1])
        subsample = aux[aux['label'] == 0].sample(len_positive)
        subsample = subsample.append(aux[aux['label'] == 1])
        print("Distribuição dados de treinamento depois do balanceamento:", subsample['label'].value_counts())
        X = subsample['episode'].tolist()
        classes = subsample['label'].tolist()
        #############################################
        ### Balancing instances on optimization data ###
        #############################################
        aux = data_csv[data_csv['episode'].isin(data_opt)]
        len_positive = len(aux[aux['label'] == 1])
        subsample = aux[aux['label'] == 0].sample(len_positive)
        subsample = subsample.append(aux[aux['label'] == 1])
        print("Distribuição dados de otimização depois do balanceamento:", subsample['label'].value_counts())
        X_opt = subsample['episode'].tolist()
        classes_opt = subsample['label'].tolist()

    with open(training_samples_path, 'wb') as f:
        pickle.dump(data, f)
    with open(training_classes_path, 'wb') as f:
        pickle.dump(classes, f)
    with open(optimization_samples_path, 'wb') as f :
        pickle.dump(data_opt, f)
    with open(optimization_classes_path, 'wb') as f:
        pickle.dump(classes_opt, f)
else:
    with open(training_samples_path, 'rb') as f:
        data = pickle.load(f)
    with open(training_classes_path, 'rb') as f:
        classes = pickle.load(f)
    with open(optimization_samples_path, 'rb') as f :
        data_opt = pickle.load(f)
    with open(optimization_classes_path, 'rb') as f:
        classes_opt = pickle.load(f)

optimization_df = data_csv[data_csv['episode'].isin(data_opt)]
classes_opt = np.asarray(optimization_df['label'].tolist())
optimization_data = np.asarray(optimization_df['path'].tolist())
opt_normalization_values_path = training_directory + parameters['optimization_normalization_values_filename']

print_with_time("Creating optimization generators")
training_events_sizes_file_path = training_directory + parameters['training_events_sizes_filename'].format('opt')
training_events_sizes_labels_file_path = training_directory + parameters[
    'training_events_sizes_labels_filename'].format('opt')

train_sizes, train_labels = functions.divide_by_events_lenght(optimization_data
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


data = data_csv[data_csv['episode'].isin(data)]
classes = np.asarray(data['label'].tolist())
data = np.asarray(data['path'].tolist())

# Using a seed always will get the same data split even if the training stops
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=15)
fold = 0
# ====================== Script that start training new models
result_file_path = checkpoint_directory + parameters['result_filename']
with open(result_file_path, 'a+') as cvsFileHandler: # where the results for each fold are appended
    dictWriter = None
    eval_results = None
    for trainIndex, testIndex in kf.split(data, classes):
        trained_model_path = checkpoint_directory + parameters['trained_model_filename'].format(fold)
        if os.path.exists(trained_model_path):
            print("Pass fold {}".format(fold))
            fold += 1
            continue
        print_with_time("Fold {}".format(fold))
        training_events_sizes_file_path = training_directory + parameters['training_events_sizes_filename'].format(fold)
        training_events_sizes_labels_file_path = training_directory + parameters['training_events_sizes_labels_filename'].format(fold)
        testing_events_sizes_file_path = training_directory + parameters['testing_events_sizes_filename'].format(fold)
        testing_events_sizes_labels_file_path = training_directory + parameters['testing_events_sizes_labels_filename'].format(fold)

        train_sizes, train_labels = functions.divide_by_events_lenght(data[trainIndex]
                                                                      , classes[trainIndex]
                                                                      , sizes_filename=training_events_sizes_file_path
                                                                      , classes_filename=training_events_sizes_labels_file_path)
        test_sizes, test_labels = functions.divide_by_events_lenght(data[testIndex], classes[testIndex]
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
        fold += 1

# Evaluating k-fold
results = pd.read_csv(result_file_path)
print(results.describe())
