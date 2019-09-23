import os
from functools import partial

import numpy
import pandas as pd
import multiprocessing as mp

import sys
from nltk import WhitespaceTokenizer

import functions
import re


def get_replacements(tokens):
    replacements = dict()
    for token in tokens:
        if re.match("\d*-\d*-\d*", token):
            replacement = "[**date**]"
        elif re.match("\d*-\d*", token):
            replacement = ""
        elif re.match("^[0-9 ]+$", token.strip()):
            replacement = "[**number**]"
        else:
            replacement = re.sub("[\(\[].*?[\)\]]", "", token)
            replacement = replacement.lower().replace(" ", "")
            replacement = ''.join(e for e in replacement if e.isalpha())
        replacements["[**{}**]".format(token)] = "[**{}**]".format(replacement)
    return replacements


def process_notes(icustays, noteevents_data_path=None, new_noteevents_path=None, manager_queue=None):
    for icustay in icustays:
        if manager_queue is not None:
            manager_queue.put(icustay)
        if not os.path.exists(noteevents_data_path + "{}.csv".format(icustay)):
            continue
        patient_noteevents = pd.read_csv(noteevents_data_path + "{}.csv".format(icustay))
        new_noteevents = pd.DataFrame([])
        for index, row in patient_noteevents.iterrows():
            tokens = re.findall(r'\[\*\*(.*?)\*\*\]', row['Note'])
            replacements = get_replacements(tokens)
            for token, replacement in replacements.items():
                row['Note'] = row['Note'].replace(token, replacement)
            new_noteevents.append(row)
        new_noteevents.to_csv(new_noteevents_path + "{}.csv".format(icustay))


parameters = functions.load_parameters_file()
dataset = pd.read_csv(parameters['mimic_data_path'] + parameters['dataset_file_name'])
noteevents_data_path = parameters['mimic_data_path'] + "sepsis_noteevents/"
icustays = dataset['icustay_id'].tolist()
icustays = numpy.array_split(icustays, 10)
if not os.path.exists(parameters['mimic_data_path'] + parameters['noteevents_anonymized_tokens_normalized']):
    os.mkdir(parameters['mimic_data_path'] + parameters['noteevents_anonymized_tokens_normalized'])

with mp.Pool(processes=4) as pool:
    m = mp.Manager()
    queue = m.Queue()
    partial_process_notes = partial(process_notes,
                                    noteevents_data_path=noteevents_data_path,
                                    new_noteevents_path=parameters['mimic_data_path']
                                                        + parameters['noteevents_anonymized_tokens_normalized'],
                                    manager_queue=queue)
    print("===== Processing events =====")
    map_obj = pool.map_async(partial_process_notes, icustays)
    consumed = 0
    while not map_obj.ready():
        for _ in range(queue.qsize()):
            queue.get()
            consumed += 1
        sys.stderr.write('\rdone {0:%}'.format(consumed / len(dataset)))
