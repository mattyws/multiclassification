import os
from functools import partial

import numpy
import pandas as pd
import multiprocessing as mp

import sys
from nltk import WhitespaceTokenizer

import functions
import re

def process_notes(icustays, noteevents_data_path=None, manager_queue=None):
    anonymized_tokens = set()
    for icustay in icustays:
        if manager_queue is not None:
            manager_queue.put(icustay)
        if not os.path.exists(noteevents_data_path + "{}.csv".format(icustay)):
            continue
        patient_noteevents = pd.read_csv(noteevents_data_path + "{}.csv".format(icustay))
        for index, row in patient_noteevents.iterrows():
            tokens = re.findall(r'\[(.*?)\]', row['Note'])
            anonymized_tokens.update(tokens)
            if "**MD Number(3) 10571**" in row['Note']:
                print(row['Note'])
    return anonymized_tokens


parameters = functions.load_parameters_file()
dataset = pd.read_csv(parameters['mimic_data_path'] + parameters['dataset_file_name'])
noteevents_data_path = parameters['mimic_data_path'] + "sepsis_noteevents/"
icustays = dataset['icustay_id'].tolist()
icustays = numpy.array_split(icustays, 10)
if not os.path.exists(parameters['mimic_data_path'] + parameters['tokenized_noteevents_dirname']):
    os.mkdir(parameters['mimic_data_path'] + parameters['tokenized_noteevents_dirname'])

with mp.Pool(processes=4) as pool:
    m = mp.Manager()
    queue = m.Queue()
    partial_process_notes = partial(process_notes,
                                    noteevents_data_path=noteevents_data_path,
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
    anonymized_tokens = set()
    for result in results:
        anonymized_tokens.update(result)
    anonymized_tokens = list(anonymized_tokens)
    anonymized_tokens.sort()
    with open('anonymized_tokens', 'w') as f:
        for token in anonymized_tokens:
            f.write(token+'\n')
