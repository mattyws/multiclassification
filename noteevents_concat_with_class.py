import csv
import html
import json
import logging
import os

import sys
from ast import literal_eval

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""

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
new_representation_path = "/home/mattyws/Documents/mimic/concateneted_words_and_class.csv"
new_data = []
total_files = len(data)
consumed = -1
for icustay, clas in zip(data, classes):
    consumed += 1
    sys.stderr.write('\rdone {0:%}'.format(consumed / total_files))
    concat_text = ""
    icustay_id = icustay.split('/')[-1].split('.')[0]
    events = pd.read_csv(icustay)
    events = events.replace(np.nan, '')
    for event in events['words']:
        event = literal_eval(event)
        concat_text += ' '.join(event)
    icustay_concat = dict()
    icustay_concat["words"] = concat_text
    icustay_concat["class"] = clas
    icustay_concat["icustay_id"] = icustay_id
    new_data.append(icustay_concat)

new_data = pd.DataFrame(new_data)
new_data.to_csv(new_representation_path, index=False)
