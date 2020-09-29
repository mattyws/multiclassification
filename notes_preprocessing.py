import multiprocessing
from functools import partial

import numpy
import pandas as pd
import os
import sys
from resources.functions import test_model, print_with_time, escape_invalid_xml_characters, escape_html_special_entities, \
    text_to_lower, remove_sepsis_mentions, remove_only_special_characters_tokens, whitespace_tokenize_text, \
    train_representation_model

new_representation_path = "/home/mattyws/Documents/mimic/textual_normalized_preprocessed/"
if not os.path.exists(new_representation_path):
    os.mkdir(new_representation_path)

def transform_representations(icustays, new_representation_path=None, manager_queue=None):
    for icustay in icustays:
        if os.path.exists(new_representation_path + "{}.csv".format(icustay)):
            continue
        if manager_queue is not None:
            manager_queue.put(icustay)
        textual_data = pd.read_csv("/home/mattyws/Documents/mimic/textual_anonymized_data/{}.csv".format(icustay))
        preprocessed_texts = []
        for text_index, text_row in textual_data.iterrows():
            text = text_row['Note']
            for func in preprocessing_pipeline:
                text = func(text)
            preprocessed_texts.append(text)
        textual_data['preprocessed_note'] = [' '.join(text) for text in preprocessed_texts]
        textual_data['charttime'] = textual_data['Unnamed: 0']
        textual_data = textual_data.drop(columns=['Unnamed: 0'])
        textual_data.to_csv(new_representation_path + "{}.csv".format(icustay), index=False)

preprocessing_pipeline = [escape_invalid_xml_characters, escape_html_special_entities, text_to_lower,
                              whitespace_tokenize_text, remove_only_special_characters_tokens, remove_sepsis_mentions]
print_with_time("Loading data")
data_csv = pd.read_csv("/home/mattyws/Documents/mimic/new_dataset_patients.csv")
icustays = data_csv['icustay_id'].tolist()

with multiprocessing.Pool(processes=1) as pool:
    manager = multiprocessing.Manager()
    manager_queue = manager.Queue()
    partial_transform_representation = partial(transform_representations,
                                               new_representation_path=new_representation_path,
                                                manager_queue=manager_queue)
    data = numpy.array_split(icustays, 6)
    # self.transform_representations(data[0], new_representation_path=new_representation_path, manager_queue=manager_queue)
    # exit()
    total_files = len(icustays)
    map_obj = pool.map_async(partial_transform_representation, data)
    consumed = 0
    while not map_obj.ready() or manager_queue.qsize() != 0:
        for _ in range(manager_queue.qsize()):
            manager_queue.get()
            consumed += 1
        sys.stderr.write('\rdone {0:%}'.format(consumed / total_files))
    result = map_obj.get()