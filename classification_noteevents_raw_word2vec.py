# TODO: change the script to use the word2vec and notes, bc now it's the copy of classification.py
import csv
import html
import json
import os
from datetime import datetime

import pandas as pd
import numpy as np

import keras
import unicodedata
from nltk import WhitespaceTokenizer

from sklearn.model_selection._split import StratifiedKFold

import functions
import train_word2vec
from data_generators import LengthLongitudinalDataGenerator
from xml.sax.saxutils import escape, quoteattr, unescape
from functions import test_model, print_with_time
from keras_callbacks import Metrics
from model_creators import MultilayerKerasRecurrentNNCreator
from normalization import Normalization, NormalizationValues

def escape_invalid_xml_characters(text):
    text = escape(text)
    text = quoteattr(text)
    text = "".join(ch for ch in text if unicodedata.category(ch)[0] != "C")
    return text


def escape_html_special_entities(text):
    return html.unescape(text)


def text_to_lower(text):
    return text.lower()


def tokenize_text(text):
    tokenizer = WhitespaceTokenizer()
    return tokenizer.tokenize(text)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
DATETIME_PATTERN = "%Y-%m-%d %H:%M:%S"

parametersFilePath = "./classification_noteevents_word2vec_parameters.json"

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

# TODO: Transfer these variables to a .json parameter file
embedding_size = 400
min_count = 2
workers = 4
window = 2
iterations = 30


inputShape = (None, embedding_size)
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
        print_with_time("Training Word2vec")
        preprocessing_pipeline = [escape_invalid_xml_characters, escape_html_special_entities, text_to_lower, tokenize_text,
                                  functions.remove_only_special_characters_tokens]
        wor2vec_model = train_word2vec.train(data[trainIndex],
                                             parameters['word2vecModelFileName'].format(i), min_count,
                                             embedding_size, workers, window, iterations,
                                             preprocessing_pipeline=preprocessing_pipeline)
        # TODO: use the word2vec model to transform the representation of all data before creating the generator with the transformed data
        print_with_time("Transforming representation")
        for file in data:



        print_with_time("Creating generators")
        train_sizes, train_labels = functions.divide_by_events_lenght(normalized_data[trainIndex]
                                                                      , classes[trainIndex]
                                                                      , sizes_filename=parameters['training_events_sizes_file'].format(i)
                                                                      , classes_filename=parameters['training_events_sizes_labels_file'].format(i))
        test_sizes, test_labels = functions.divide_by_events_lenght(normalized_data[testIndex], classes[testIndex]
                                                            , sizes_filename = parameters['testing_events_sizes_file'].format(i)
                                                            , classes_filename = parameters['testing_events_sizes_labels_file'].format(i))
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

        modelCreator = MultilayerKerasRecurrentNNCreator(inputShape, parameters['outputUnits'], parameters['numOutputNeurons'],
                                                         loss=parameters['loss'], layersActivations=parameters['layersActivations'],
                                                         gru=parameters['gru'], use_dropout=parameters['useDropout'],
                                                         dropout=parameters['dropout'],
                                                         metrics=[keras.metrics.binary_accuracy])
        kerasAdapter = modelCreator.create(model_summary_filename=parameters['modelCheckpointPath']+'model_summary')
        epochs = parameters['trainingEpochs']
        metrics_callback = Metrics(dataTestGenerator)
        kerasAdapter.fit(dataTrainGenerator, epochs=epochs, batch_size=len(dataTrainGenerator), callbacks=[metrics_callback])
        metrics = test_model(kerasAdapter, dataTestGenerator, i)
        if dictWriter is None:
            dictWriter = csv.DictWriter(cvsFileHandler, metrics.keys())
        if metrics['fold'] == 0:
            dictWriter.writeheader()
        dictWriter.writerow(metrics)
        i += 1
