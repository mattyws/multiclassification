"""
Count number of tokens in dataset
Save: unique tokens, tokens frequency and total tokens
"""
import os
from collections import Counter

import pandas as pd
import multiprocessing as mp

import nltk
from nltk import WhitespaceTokenizer
from nltk.tokenize.api import StringTokenizer

import functions

def create_features_dict():
    return {
    "tokens_frequency" : dict(),
    "unique_tokens" : set(),
    "total_tokens" : 0
    }


def get_tokens_frequency(tokens, frequencies=None):
    if frequencies is None:
        frequencies = dict()
    counter = Counter(tokens)
    frequencies = {k: frequencies.get(k, 0) + counter.get(k, 0) for k in set(frequencies) | set(counter)}
    return frequencies


def get_unique_tokens(tokens, unique_tokens=None):
    if unique_tokens is None:
        unique_tokens = set()
    unique_tokens.update(set(tokens))
    return unique_tokens


def get_total_tokens(tokens, total_tokens=None):
    if total_tokens is None:
        total_tokens = 0
    total_tokens += len(tokens)
    return total_tokens


def get_tokens_features(tokens, tokens_features=None):
    if tokens_features is None:
        tokens_features = create_features_dict()
    tokens_features['tokens_frequency'] = get_tokens_frequency(tokens, frequencies=tokens_features['tokens_frequency'])
    tokens_features['unique_tokens'] = get_unique_tokens(tokens, unique_tokens=tokens_features['unique_tokens'])
    tokens_features['total_tokens'] = get_total_tokens(tokens, total_tokens=tokens_features['total_tokens'])
    return tokens_features

parameters = functions.load_parameters_file()
dataset = pd.read_csv(parameters['mimic_data_path'] + parameters['dataset_file_name'])
noteevents_data_path = parameters['mimic_data_path'] + "sepsis_noteevents/"

tokenizer = WhitespaceTokenizer()
icustays = dataset['icustay_id'].tolist()
tokens_features = create_features_dict()

for icustay in icustays:
    if not os.path.exists(noteevents_data_path + "{}.csv".format(icustay)):
        continue
    patient_noteevents = pd.read_csv(noteevents_data_path + "{}.csv".format(icustay))
    notes = patient_noteevents['Note'].tolist()
    patient_tokens = []
    for note in notes:
        tokens = tokenizer.tokenize(note)
        tokens = [word.lower() for word in tokens]
        patient_tokens.extend(tokens)
    tokens_features = None
    exit()
