import csv
import html
import json
import logging
import os

import pandas as pd
import numpy as np

import keras
from keras.regularizers import l1_l2

from sklearn.model_selection._split import StratifiedKFold, train_test_split

from adapter import KerasAdapter
from data_generators import LengthLongitudinalDataGenerator, BertDataGenerator

from data_representation import TransformClinicalTextsRepresentations, TextToBertIDs
from functions import test_model, print_with_time, escape_invalid_xml_characters, escape_html_special_entities, \
    text_to_lower, remove_only_special_characters_tokens, whitespace_tokenize_text, \
    divide_by_events_lenght, remove_sepsis_mentions, train_representation_model
from keras_callbacks import Metrics
from model_creators import MultilayerTemporalConvolutionalNNCreator, BertModelCreator

from classification_noteevents_textual_parameters import parameters

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
def sync_data_classes(data, classes):
    new_dataset = []
    new_classes = []
    for d, c in zip(data, classes):
        if d is not None:
            new_dataset.append(d)
            new_classes.append(c)
    return np.array(new_dataset), np.array(new_classes)

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
DATETIME_PATTERN = "%Y-%m-%d %H:%M:%S"

parametersFilePath = "./classification_noteevents_textual_parameters.py"

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
# Using a seed always will get the same data split even if the training stops

X_train, data, y_train, classes = train_test_split(data, classes, test_size=0.20, random_state=15, stratify=classes)


kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=15)

embedding_size = parameters['embedding_size']
min_count = parameters['min_count']
workers = parameters['workers']
window = parameters['window']
iterations = parameters['iterations']
inputShape = (None, embedding_size)

# TODO: remove training with doc2vec, and use the same logic with albert, but for each fold (?)
# print_with_time("Training/Loading Doc2Vec")
# preprocessing_pipeline = [escape_invalid_xml_characters, escape_html_special_entities, text_to_lower,
#                           whitespace_tokenize_text, remove_only_special_characters_tokens, remove_sepsis_mentions]
# doc2vec_model = train_representation_model(data, parameters['word2vecModelFileName'], min_count,
#                                            embedding_size, workers, window, iterations,
#                                            preprocessing_pipeline=preprocessing_pipeline, word2vec=False)
# print_with_time("Transforming/Retrieving representation")
# texts_transformer = TransformClinicalTextsRepresentations(doc2vec_model, embedding_size=embedding_size,
#                                                           window=window, texts_path=parameters['dataPath'],
#                                                           representation_save_path=parameters['word2vec_representation_files_path'],
#                                                           is_word2vec=False)
# doc2vec_model = None
# texts_transformer.transform(data, preprocessing_pipeline=preprocessing_pipeline)
# normalized_data = np.array(texts_transformer.get_new_paths(data))
# normalized_data, classes = sync_data_classes(normalized_data, classes)

text_to_ids = TextToBertIDs(data, model_dir="/home/mattyws/Downloads/albert_base/")
text_to_ids.transform("/home/mattyws/Documents/mimic/bert_last_ids/")
data = np.array(text_to_ids.get_new_paths(data))

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
        if os.path.exists("../mimic/albert_raw_training/checkpoint/albert_model_{}.model".format(i)):
            adapter = KerasAdapter.load_model("../mimic/albert_raw_training/checkpoint/albert_model_{}.model".format(i))
        else:
            train_generator = BertDataGenerator(data[trainIndex], classes[trainIndex], 8)
            test_generator = BertDataGenerator(data[testIndex], classes[testIndex], 8)
            # for index in range(len(train_generator)):
            #     print(train_generator[index][0].shape)
            #     exit()
            model_generator = BertModelCreator((None,512))
            model, l_bert = model_generator.create_representation_model()
            model.fit_generator(generator=train_generator,
                                epochs=3
                                # callbacks=[
                                #     keras.callbacks.EarlyStopping(monitor='val_sparse_categorical_accuracy', patience=5),
                                # ],
                                )
            adapter = KerasAdapter(model)
            print("Saving models")
            adapter.save("../mimic/albert_raw_training/checkpoint/albert_model_{}.model".format(i))
        metrics = test_model(adapter, test_generator, i)
        if dictWriter is None:
            dictWriter = csv.DictWriter(cvsFileHandler, metrics.keys())
        if metrics['fold'] == 0:
            dictWriter.writeheader()
        dictWriter.writerow(metrics)
        i += 1
