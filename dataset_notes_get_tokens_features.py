"""
Count number of tokens in dataset
Save: unique tokens, tokens frequency and total tokens
"""
import os
import pickle
from collections import Counter
from functools import partial

import numpy
import pandas as pd
import multiprocessing as mp

import nltk
import sys
from nltk import WhitespaceTokenizer
from nltk.tokenize.api import StringTokenizer

import functions

def create_features_dict():
    return {
    "tokens_frequency" : dict(),
    "unique_tokens" : set(),
    "total_tokens" : 0
    }

def merge_frequencies(dict1, dict2):
    return {k: dict1.get(k, 0) + dict2.get(k, 0) for k in set(dict1) | set(dict2)}

def get_tokens_frequency(tokens, frequencies=None):
    if frequencies is None:
        frequencies = dict()
    counter = Counter(tokens)
    frequencies = merge_frequencies(frequencies, counter)
    return frequencies

def merge_uniques_tokens(set1, set2):
    set1.update(set2)
    return set1

def get_unique_tokens(tokens, unique_tokens=None):
    if unique_tokens is None:
        unique_tokens = set()
    unique_tokens = merge_uniques_tokens(unique_tokens, tokens)
    return unique_tokens

def merge_total_tokens(value1, value2):
    return value1 + value2

def get_total_tokens(tokens, total_tokens=None):
    if total_tokens is None:
        total_tokens = 0
    total_tokens = merge_total_tokens(total_tokens, len(tokens))
    return total_tokens

def merge_features_dictionaries(features1, features2):
    merged_features = create_features_dict()
    merged_features['tokens_frequency'] = merge_frequencies(features1['tokens_frequency'], features2['tokens_frequency'])
    merged_features['unique_tokens']  = merge_uniques_tokens(features1['unique_tokens'], features2['unique_tokens'])
    'total_tokens': 0

def get_tokens_features(tokens, tokens_features=None):
    if tokens_features is None:
        tokens_features = create_features_dict()
    tokens_features['tokens_frequency'] = get_tokens_frequency(tokens, frequencies=tokens_features['tokens_frequency'])
    tokens_features['unique_tokens'] = get_unique_tokens(tokens, unique_tokens=tokens_features['unique_tokens'])
    tokens_features['total_tokens'] = get_total_tokens(tokens, total_tokens=tokens_features['total_tokens'])
    return tokens_features

def process_notes(icustays, manager_queue=None):
    tokenizer = WhitespaceTokenizer()
    tokens_features = create_features_dict()
    for icustay in icustays:
        if manager_queue is not None:
            manager_queue.put(icustay)
        if not os.path.exists(noteevents_data_path + "{}.csv".format(icustay)):
            continue
        patient_noteevents = pd.read_csv(noteevents_data_path + "{}.csv".format(icustay))
        notes = patient_noteevents['Note'].tolist()
        patient_tokens = []
        for note in notes:
            tokens = tokenizer.tokenize(note)
            tokens = [word.lower() for word in tokens]
            patient_tokens.extend(tokens)
        tokens_features = get_tokens_features(patient_tokens, tokens_features=tokens_features)
    return tokens_features

parameters = functions.load_parameters_file()
dataset = pd.read_csv(parameters['mimic_data_path'] + parameters['dataset_file_name'])
noteevents_data_path = parameters['mimic_data_path'] + "sepsis_noteevents/"
icustays = dataset['icustay_id'].tolist()
icustays = numpy.array_split(icustays, 10)

with mp.Pool(processes=4) as pool:
    m = mp.Manager()
    queue = m.Queue()
    partial_process_notes = partial(process_notes,
                                    manager_queue=queue)
    print("===== Processing events =====")
    print()
    map_obj = pool.map_async(partial_process_notes, icustays)
    consumed = 0
    while not map_obj.ready():
        for _ in range(queue.qsize()):
            queue.get()
            consumed += 1
        sys.stderr.write('\rdone {0:%}'.format(consumed / len(dataset)))
    results = map_obj.get()
    tokens_features = create_features_dict()
    print("===== Merging results =====")
    for result in results:


    with open(parameters['mimic_data_path'] + parameters['notes_tokens_features_file_name'], 'wb') as file:
        pickle.dump(tokens_features, file)
    print(tokens_features['total_tokens'])
