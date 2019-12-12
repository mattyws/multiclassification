import csv
import html
import json
import logging
import os
from datetime import datetime

import pandas as pd
import numpy as np

import keras

from sklearn.model_selection._split import StratifiedKFold

from adapter import Word2VecTrainer
from data_generators import LengthLongitudinalDataGenerator, NoteeventsTextDataGenerator

from data_representation import TransformClinicalTextsRepresentations
from functions import test_model, print_with_time, escape_invalid_xml_characters, escape_html_special_entities, \
    text_to_lower, tokenize_text, remove_only_special_characters_tokens, whitespace_tokenize_text, \
    divide_by_events_lenght, remove_sepsis_mentions
from keras_callbacks import Metrics
from model_creators import MultilayerKerasRecurrentNNCreator, NoteeventsClassificationModelCreator

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


def train_word2vec(files_paths, saved_model_path, min_count, size, workers, window, iterations, noteevents_iterator=None, preprocessing_pipeline=None):
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    if noteevents_iterator is None:
        noteevents_iterator = NoteeventsTextDataGenerator(files_paths, preprocessing_pipeline=preprocessing_pipeline)
    # for noteevent in noteevents_iterator:
    #     print(noteevent)
    #     exit()
    word2vec_trainer = Word2VecTrainer(min_count=min_count, size=size, workers=workers, window=window, iter=iterations)
    if os.path.exists(saved_model_path):
        model = word2vec_trainer.load_model(saved_model_path)
        return model
    else:
        word2vec_trainer.train(noteevents_iterator)
        word2vec_trainer.save(saved_model_path)
        return word2vec_trainer.model

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
word2vec_data = np.array([parameters['notes_word2vec_path'] + '{}.txt'.format(itemid) for itemid in data])
data = np.array([parameters['dataPath'] + '{}.csv'.format(itemid) for itemid in data])
print("========= Transforming classes")
classes = np.array([1 if c == 'sepsis' else 0 for c in list(data_csv['class'])])
classes_for_stratified = np.array([1 if c == 'sepsis' else 0 for c in list(data_csv['class'])])
# Using a seed always will get the same data split even if the training stops
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=15)

embedding_size = parameters['embedding_size']
min_count = parameters['min_count']
workers = parameters['workers']
window = parameters['window']
iterations = parameters['iterations']
inputShape = (None, None, embedding_size)

print_with_time("Training Word2vec")
preprocessing_pipeline = [escape_invalid_xml_characters, escape_html_special_entities, text_to_lower,
                          whitespace_tokenize_text, remove_only_special_characters_tokens, remove_sepsis_mentions]
word2vec_model = train_word2vec(word2vec_data,
                                parameters['word2vecModelFileName'], min_count,
                                embedding_size, workers, window, iterations)
print_with_time("Transforming representation")
texts_transformer = TransformClinicalTextsRepresentations(word2vec_model, embedding_size=embedding_size,
                                                          window=window, texts_path=parameters['dataPath'],
                                                          representation_save_path=parameters['word2vec_representation_files_path'])
word2vec_model = None
texts_transformer.transform(data, preprocessing_pipeline=preprocessing_pipeline)
normalized_data = np.array(texts_transformer.get_new_paths(data))

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
        print_with_time("Creating generators")
        train_sizes, train_labels = divide_by_events_lenght(normalized_data[trainIndex]
                                                                      , classes[trainIndex]
                                                                      , sizes_filename=parameters['training_events_sizes_file'].format(i)
                                                                      , classes_filename=parameters['training_events_sizes_labels_file'].format(i))
        test_sizes, test_labels = divide_by_events_lenght(normalized_data[testIndex], classes[testIndex]
                                                            , sizes_filename = parameters['testing_events_sizes_file'].format(i)
                                                            , classes_filename = parameters['testing_events_sizes_labels_file'].format(i))
        dataTrainGenerator = LengthLongitudinalDataGenerator(train_sizes, train_labels)
        dataTrainGenerator.create_batches()
        dataTestGenerator = LengthLongitudinalDataGenerator(test_sizes, test_labels)
        dataTestGenerator.create_batches()
        print(test_sizes)
        for i in range(0, len(dataTestGenerator)):
            test = np.array(dataTestGenerator[i][0])
            # print(test.shape)
            print(test[0].shape)
            # print(test[0])
            for note in test[0]:
                print(note)
                print(note.shape)
                print(note[0].shape)
            # print(test)
            exit()
        # dataTrainGenerator = LongitudinalDataGenerator(normalized_data[trainIndex],
        #                                                classes[trainIndex], parameters['batchSize'],
        #                                                saved_batch_dir='training_batches_fold_{}'.format(i))
        # dataTestGenerator = LongitudinalDataGenerator(normalized_data[testIndex],
        #                                               classes[testIndex], parameters['batchSize'],
        #                                               saved_batch_dir='testing_batches_fold_{}'.format(i))

        modelCreator = NoteeventsClassificationModelCreator(inputShape, parameters['outputUnits'], parameters['numOutputNeurons'],
                                                         embedding_size=embedding_size,
                                                         loss=parameters['loss'], layersActivations=parameters['layersActivations'],
                                                         gru=parameters['gru'], use_dropout=parameters['useDropout'],
                                                         dropout=parameters['dropout'],
                                                         metrics=[keras.metrics.binary_accuracy])
        kerasAdapter = modelCreator.create(model_summary_filename=parameters['modelCheckpointPath']+'model_summary')
        epochs = parameters['trainingEpochs']
        metrics_callback = Metrics(dataTestGenerator)
        print_with_time("Training model")
        kerasAdapter.fit(dataTrainGenerator, epochs=epochs, batch_size=len(dataTrainGenerator))
        print_with_time("Testing model")
        metrics = test_model(kerasAdapter, dataTestGenerator, i)
        if dictWriter is None:
            dictWriter = csv.DictWriter(cvsFileHandler, metrics.keys())
        if metrics['fold'] == 0:
            dictWriter.writeheader()
        dictWriter.writerow(metrics)
        i += 1
