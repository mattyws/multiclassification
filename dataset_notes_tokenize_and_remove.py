import os
from functools import partial

import numpy
import pandas as pd
from nltk import WhitespaceTokenizer

import functions
import re


def remove_only_special_characters_tokens(tokens):
    new_tokens = []
    for token in tokens:
        if not re.match(r'^[_\W]+$', token):
            new_tokens.append(token)
    return new_tokens


def process_notes(icustays, noteevents_data_path=None, tokenized_events_data_path=None, manager_queue=None):
    tokenizer = WhitespaceTokenizer()
    for icustay in icustays:
        if manager_queue is not None:
            manager_queue.put(icustay)
        if not os.path.exists(noteevents_data_path + "{}.csv".format(icustay)):
            continue
        patient_noteevents = pd.read_csv(noteevents_data_path + "{}.csv".format(icustay))
        for index, row in patient_noteevents.iterrows():
            tokens = tokenizer.tokenize(row['Note'])
            tokens = [word.lower() for word in tokens]
            tokens = remove_only_special_characters_tokens(tokens)
            #TODO create new csv file with new tokens
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
        tokens_features = merge_features_dictionaries(tokens_features, result)

    with open(parameters['mimic_data_path'] + parameters['notes_tokens_features_file_name'], 'wb') as file:
        pickle.dump(tokens_features, file)
    print(tokens_features['total_tokens'])
