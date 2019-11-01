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


def train(files_paths, saved_model_path, min_count, size, workers, window, iterations, noteevents_iterator=None, preprocessing_pipeline=None):
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


if __name__ == "__main__":

    parameters = functions.load_parameters_file()
    dataset = pd.read_csv(parameters['mimic_data_path'] + parameters['dataset_file_name'])
    noteevents_data_path = parameters['mimic_data_path'] + parameters['tokenized_noteevents_dirname']
    icustays = dataset['icustay_id'].tolist()
    saved_model_path = parameters['mimic_data_path'] + parameters['word2vec_model_file_name']

    files_paths = [noteevents_data_path + "{}.csv".format(x) for x in icustays if os.path.exists(noteevents_data_path + "{}.csv".format(x))]

    # Parameters
    min_count=2
    size=300
    workers=4
    window=3
    iterations=10
    train(files_paths, saved_model_path, min_count, size, workers, window, iterations)

