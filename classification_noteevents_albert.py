import csv
import html
import json
import logging
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import pandas as pd
import numpy as np

from sklearn.model_selection._split import StratifiedKFold, train_test_split

from adapter import KerasAdapter
from data_generators import LengthLongitudinalDataGenerator, BertDataGenerator

from data_representation import TransformClinicalTextsRepresentations, TextToBioBertIDs
from functions import test_model, print_with_time, escape_invalid_xml_characters, escape_html_special_entities, \
    text_to_lower, remove_only_special_characters_tokens, whitespace_tokenize_text, \
    divide_by_events_lenght, remove_sepsis_mentions, train_representation_model
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
                 if os.path.exists("../mimic/textual_normalized_preprocessed/" + '{}.csv'.format(itemid))])
data_csv = data_csv[data_csv['icustay_id'].isin(data)]
data = np.array(["../mimic/textual_normalized_preprocessed/" + '{}.csv'.format(itemid) for itemid in data])
print("========= Transforming classes")
classes = np.array([1 if c == 'sepsis' else 0 for c in list(data_csv['class'])])
# Using a seed always will get the same data split even if the training stops

# X_train, data, y_train, classes = train_test_split(data, classes, test_size=0.20, random_state=15, stratify=classes)


kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=15)

embedding_size = parameters['embedding_size']
min_count = parameters['min_count']
workers = parameters['workers']
window = parameters['window']
iterations = parameters['iterations']
inputShape = (None, embedding_size)

text_to_ids = TextToBioBertIDs(data, model_dir="../mimic/biobert_large/")
text_to_ids.transform("../mimic/biobert_last_ids/")
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
        train_generator = BertDataGenerator(data[trainIndex], classes[trainIndex], 16)
        test_generator = BertDataGenerator(data[testIndex], classes[testIndex], 16)
        if os.path.exists("../mimic/biobert_raw_training/checkpoint/albert_model_{}.model".format(i)):
            adapter = KerasAdapter.load_model("../mimic/biobert_raw_training/checkpoint/albert_model_{}.model".format(i))
        else:
            # for index in range(len(train_generator)):
            #     print(train_generator[index][0].shape)
            #     exit()
            model_generator = BertModelCreator((None,512))
            model, l_bert = model_generator.create_from_model_dir("../mimic/biobert_large/",
                                                                  "bio_bert_large_1000k.ckpt")
            model.fit_generator(generator=train_generator, epochs=3, max_queue_size=5, use_multiprocessing=True)
            adapter = KerasAdapter(model)
        metrics = test_model(adapter, test_generator, i)
        if dictWriter is None:
            dictWriter = csv.DictWriter(cvsFileHandler, metrics.keys())
        if metrics['fold'] == 0:
            dictWriter.writeheader()
        dictWriter.writerow(metrics)
        print("Saving models")
        adapter.save("../mimic/biobert_raw_training/checkpoint/albert_model_{}.model".format(i))
        i += 1
