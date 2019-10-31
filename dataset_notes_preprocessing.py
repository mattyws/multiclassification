import html
from functools import partial
from xml.sax.saxutils import escape, quoteattr

import pandas
import numpy as np
import multiprocessing as mp
import nltk
import os

import sys

import unicodedata

from nltk import WhitespaceTokenizer

import functions
from functions import print_with_time


def escape_invalid_xml_characters(text):
    text = escape(text)
    text = quoteattr(text)
    text = "".join(ch for ch in text if unicodedata.category(ch)[0] != "C")
    return text


def escape_html_special_entities(text):
    return html.unescape(text)


def text_to_lower(text):
    return text.lower()


def note_preprocessing(icustays, preprocessing_pipeline, noteevents_path=None, preprocessed_data_path=None, manager_queue=None):
    for icustay in icustays:
        if manager_queue is not None:
            manager_queue.put(icustay)
        if not os.path.exists(noteevents_path+str(icustay)+'.csv') or \
                os.path.exists(preprocessed_data_path + str(icustay) + '.csv'):
            continue
        new_events = []
        icu_noteevents = pandas.read_csv(noteevents_path+str(icustay)+'.csv')
        for index, noteevent in icu_noteevents.iterrows():
            event = dict()
            text = noteevent['Note']
            for func in preprocessing_pipeline:
                text = func(text)
            event['timestamp'] = noteevent['Unnamed: 0']
            event['Note'] = text
            new_events.append(event)
        new_events = pandas.DataFrame(new_events)
        new_events.to_csv(preprocessed_data_path + str(icustay) + '.csv', index=False)

def tokenize_text(text):
    tokenizer = WhitespaceTokenizer()
    return tokenizer.tokenize(text)

parameters = functions.load_parameters_file()

dataset_csv = pandas.read_csv(parameters['mimic_data_path'] + parameters['dataset_file_name'])
noteevents_path = parameters['mimic_data_path'] + parameters['noteevents_anonymized_tokens_normalized']
icustays = dataset_csv['icustay_id']
icustays = np.array_split(icustays, 10)
preprocessed_noteevents_path = parameters['mimic_data_path'] + 'sepsis_noteevents_preprocessed/'

preprocessing_pipeline = [escape_invalid_xml_characters, escape_html_special_entities, text_to_lower, tokenize_text,
                                  functions.remove_only_special_characters_tokens]
if not os.path.exists(preprocessed_noteevents_path):
    os.mkdir(preprocessed_noteevents_path)
with mp.Pool(processes=4) as pool:
    m = mp.Manager()
    queue = m.Queue()
    partial_preprocessing_noteevents = partial(note_preprocessing,
                                               preprocessing_pipeline=preprocessing_pipeline,
                                               noteevents_path = noteevents_path,
                                               preprocessed_data_path=preprocessed_noteevents_path,
                                               manager_queue=queue)
    partial_preprocessing_noteevents(icustays[0])
    exit()
    print_with_time("Preprocessing noteevents")
    map_obj = pool.map_async(partial_preprocessing_noteevents, icustays)
    consumed = 0
    while not map_obj.ready() or queue.qsize() != 0:
        for _ in range(queue.qsize()):
            queue.get()
            consumed += 1
        sys.stderr.write('\rdone {0:%}'.format(consumed / len(dataset_csv)))