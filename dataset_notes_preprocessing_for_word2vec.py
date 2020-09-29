from functools import partial

import pandas
import numpy as np
import multiprocessing as mp
import os

import sys
from resources.functions import print_with_time, remove_only_special_characters_tokens, escape_invalid_xml_characters, \
    escape_html_special_entities, text_to_lower, tokenize_text, tokenize_sentences, load_parameters_file


def note_preprocessing(icustays, noteevents_path=None, preprocessed_data_path=None, manager_queue=None):
    for icustay in icustays:
        if manager_queue is not None:
            manager_queue.put(icustay)
        if not os.path.exists(noteevents_path+str(icustay)+'.csv') or \
                os.path.exists(preprocessed_data_path + str(icustay) + '.txt'):
            continue
        icu_noteevents = pandas.read_csv(noteevents_path+str(icustay)+'.csv')
        icu_sentences = []
        for index, noteevent in icu_noteevents.iterrows():
            text = noteevent['Note']
            text = escape_invalid_xml_characters(text)
            text = escape_html_special_entities(text)
            text = text_to_lower(text)
            text = tokenize_text(text)
            text = tokenize_sentences(text)
            for sentence in text:
                sentence = ' '.join(remove_only_special_characters_tokens(sentence)).replace('"', '')
                if len(sentence.strip()) != 0:
                    icu_sentences.append(sentence)
        with open(preprocessed_data_path + str(icustay) + '.txt', 'w') as handler:
            for sentence in icu_sentences:
                handler.write(sentence + '\n')


parameters = load_parameters_file()

dataset_csv = pandas.read_csv(parameters['mimic_data_path'] + parameters['dataset_file_name'])
noteevents_path = parameters['mimic_data_path'] + parameters['noteevents_anonymized_tokens_normalized']
icustays = dataset_csv['icustay_id']
icustays = np.array_split(icustays, 10)
preprocessed_noteevents_path = parameters['mimic_data_path'] + 'sepsis_noteevents_preprocessed/'

preprocessing_pipeline = [escape_invalid_xml_characters, escape_html_special_entities, text_to_lower, tokenize_text,
                                  remove_only_special_characters_tokens]
if not os.path.exists(preprocessed_noteevents_path):
    os.mkdir(preprocessed_noteevents_path)
with mp.Pool(processes=4) as pool:
    m = mp.Manager()
    queue = m.Queue()
    partial_preprocessing_noteevents = partial(note_preprocessing,
                                               noteevents_path = noteevents_path,
                                               preprocessed_data_path=preprocessed_noteevents_path,
                                               manager_queue=queue)
    print_with_time("Preprocessing noteevents")
    map_obj = pool.map_async(partial_preprocessing_noteevents, icustays)
    consumed = 0
    while not map_obj.ready() or queue.qsize() != 0:
        for _ in range(queue.qsize()):
            queue.get()
            consumed += 1
        sys.stderr.write('\rdone {0:%}'.format(consumed / len(dataset_csv)))