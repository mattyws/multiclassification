"""
Train a word2vec model using Notes from EHR.
"""
import logging
import os

import numpy
import pandas as pd

import functions
from adapter import Word2VecTrainer
from data_generators import NoteeventsTextDataGenerator

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

parameters = functions.load_parameters_file()

parameters = functions.load_parameters_file()
dataset = pd.read_csv(parameters['mimic_data_path'] + parameters['dataset_file_name'])
noteevents_data_path = parameters['mimic_data_path'] + parameters['tokenized_noteevents_dirname']
icustays = dataset['icustay_id'].tolist()

files_paths = [noteevents_data_path + "{}.csv".format(x) for x in icustays if os.path.exists(noteevents_data_path + "{}.csv".format(x))]

# Parameters
min_count=2
size=300
workers=4
window=3
iter=10

noteevents_iterator = NoteeventsTextDataGenerator(files_paths)
word2vec_trainer = Word2VecTrainer(min_count=min_count, size=size, workers=workers, window=window, iter=iter)
if os.path.exists(parameters['mimic_data_path'] + parameters['word2vec_model_file_name']):
    model = word2vec_trainer.load_model(parameters['mimic_data_path'] + parameters['word2vec_model_file_name'])
    print(model.wv.vocab)
else:
    word2vec_trainer.train(noteevents_iterator)
    word2vec_trainer.save(parameters['mimic_data_path'] + parameters['word2vec_model_file_name'])