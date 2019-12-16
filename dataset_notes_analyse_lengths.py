"""
Analyse the lengths of texts sequences for each patient and the size of each text
"""
import os
import pickle
from collections import Counter
from functools import partial

import numpy
import pandas as pd
import multiprocessing as mp

import sys

import functions
from functions import escape_invalid_xml_characters, escape_html_special_entities, text_to_lower, \
    remove_sepsis_mentions, remove_only_special_characters_tokens, whitespace_tokenize_text


def get_sizes(patients, events_path=None, preprocessing_pipeline=None, manager_queue=None):
    sequences_sizes = []
    texts_sizes = []
    for index, patient in patients.iterrows():
        if manager_queue is not None:
            manager_queue.put(index)
        if not os.path.exists(events_path + "{}.csv".format(patient['icustay_id'])):
            continue
        patient_events = pd.read_csv(events_path + "{}.csv".format(patient['icustay_id']))
        sequences_sizes.append(len(patient_events))
        for event_index, event in patient_events.iterrows():
            note = event['Note']
            if preprocessing_pipeline is not None:
                for f in preprocessing_pipeline:
                    note = f(note)
            texts_sizes.append(len(note))
    return sequences_sizes, texts_sizes


parameters = functions.load_parameters_file()

dataset = pd.read_csv(parameters['mimic_data_path'] + parameters['dataset_file_name'])
dataset_for_mp = numpy.array_split(dataset, 10)
preprocessing_pipeline = [escape_invalid_xml_characters, escape_html_special_entities, text_to_lower,
                          whitespace_tokenize_text, remove_only_special_characters_tokens, remove_sepsis_mentions]
total_files = len(dataset)
events_path = parameters['mimic_data_path'] + 'sepsis_noteevents/'
sequences_sizes_file_name = parameters['mimic_data_path'] + 'sequences_sizes.pkl'
texts_sizes_file_name = parameters['mimic_data_path'] + 'texts_sizes.pkl'

if os.path.exists(sequences_sizes_file_name) and os.path.exists(texts_sizes_file_name):
   with open(sequences_sizes_file_name, 'rb') as fhandler:
       sequences_sizes = pickle.load(fhandler)
   with open(texts_sizes_file_name, 'rb') as fhandler:
       texts_sizes = pickle.load(fhandler)
else:
    with mp.Pool(processes=6) as pool:
        m = mp.Manager()
        queue = m.Queue()
        partial_get_sizes = partial(get_sizes, events_path=events_path, preprocessing_pipeline=preprocessing_pipeline,
                                    manager_queue=queue)
        map_obj = pool.map_async(partial_get_sizes, dataset_for_mp)
        consumed = 0
        while not map_obj.ready():
            for _ in range(queue.qsize()):
                queue.get()
                consumed += 1
            sys.stderr.write('\rdone {0:%}'.format(consumed / total_files))
        print()
        result = map_obj.get()
        sequences_sizes = [r[0] for r in result]
        texts_sizes = [r[1] for r in result]
        sequences_sizes = numpy.concatenate(sequences_sizes, axis=0)
        texts_sizes = numpy.concatenate(texts_sizes, axis=0)
        with open(sequences_sizes_file_name, 'wb') as fhandler:
            pickle.dump(sequences_sizes, fhandler)
        with open(texts_sizes_file_name, 'wb') as fhandler:
            pickle.dump(texts_sizes, fhandler)

sequences_sizes_counter = Counter(sequences_sizes)
texts_sizes_counter = Counter(texts_sizes)
sequences_mean = numpy.mean(sequences_sizes)
sequences_std = numpy.std(sequences_sizes)
texts_mean = numpy.mean(texts_sizes)
print(sequences_sizes_counter)
print(texts_sizes_counter)
texts_std = numpy.std(texts_sizes)
print("Sequences size mean {} and std {}. The longest sequence is {} and the smallest is {}"
      .format(sequences_mean, sequences_std, max(sequences_sizes), min(sequences_sizes)))
print("Texts size mean {} and std {}. The longest sequence is {} and the smallest is {}"
      .format(texts_mean, texts_std, max(texts_sizes), min(texts_sizes)))

max_len = texts_mean + texts_std
higher_than_mean_std = [x for x in texts_sizes_counter.keys() if x > max_len]
total = 0
for value in higher_than_mean_std:
    total += texts_sizes_counter[value]
print("From a total of {} texts, {} is higher than mean + std".format(len(texts_sizes), total))

