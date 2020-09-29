import os
from functools import partial

import numpy
import pandas as pd
import multiprocessing as mp

import sys
from nltk import WhitespaceTokenizer

from resources import functions

from resources.functions import remove_only_special_characters_tokens


 def process_notes(icustays, noteevents_data_path=None, tokenized_events_data_path=None, manager_queue=None):
    tokenizer = WhitespaceTokenizer()
    for icustay in icustays:
        if manager_queue is not None:
            manager_queue.put(icustay)
        if not os.path.exists(noteevents_data_path + "{}.csv".format(icustay)):
            continue
        patient_noteevents = pd.read_csv(noteevents_data_path + "{}.csv".format(icustay))
        patient_tokenized_noteevents = pd.DataFrame([])
        for index, row in patient_noteevents.iterrows():
            tokens = tokenizer.tokenize(row['Note'])
            tokens = [word.lower() for word in tokens]
            tokens = remove_only_special_characters_tokens(tokens)
            row['Note'] = tokens
            patient_tokenized_noteevents = patient_tokenized_noteevents.append(row)
        patient_tokenized_noteevents.to_csv(tokenized_events_data_path + "{}.csv".format(icustay), index=False)

parameters = functions.load_parameters_file()
dataset = pd.read_csv(parameters['mimic_data_path'] + parameters['dataset_file_name'])
noteevents_data_path = parameters['mimic_data_path'] + parameters['noteevents_anonymized_tokens_normalized']
icustays = dataset['icustay_id'].tolist()
icustays = numpy.array_split(icustays, 10)
if not os.path.exists(parameters['mimic_data_path'] + parameters['tokenized_noteevents_dirname']):
    os.mkdir(parameters['mimic_data_path'] + parameters['tokenized_noteevents_dirname'])

with mp.Pool(processes=4) as pool:
    m = mp.Manager()
    queue = m.Queue()
    partial_process_notes = partial(process_notes,
                                    noteevents_data_path=noteevents_data_path,
                                    tokenized_events_data_path=parameters['mimic_data_path']
                                                               + parameters['tokenized_noteevents_dirname'],
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
