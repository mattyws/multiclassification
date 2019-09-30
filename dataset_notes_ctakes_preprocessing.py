import os
import subprocess
from functools import partial
from xml.sax.saxutils import escape, quoteattr

import pandas
import unicodedata
import multiprocessing as mp
import numpy as np
import sys

import functions

def escape_invalid_xml_characters(text):
    text = escape(text)
    text = quoteattr(text)
    text = "".join(ch for ch in text if unicodedata.category(ch)[0]!="C")
    return text

def split_data_for_ctakes(icustayids, noteevents_path=None, ctakes_data_path=None, manager_queue=None):
    for icustay in icustayids:
        if manager_queue is not None:
            manager_queue.put(icustay)
        if not os.path.exists(noteevents_path + "{}.csv".format(icustay)):
            continue
        icustay_path = ctakes_data_path + str(icustay) + '/'
        if not os.path.exists(icustay_path):
            os.mkdir(icustay_path)
        noteevents = pandas.read_csv(noteevents_path + "{}.csv".format(icustay))
        for index, note in noteevents.iterrows():
            new_filename = "{}_{}".format(index, note['Unnamed: 0'])
            with open(icustay_path + new_filename, 'w') as file:
                file.write(escape_invalid_xml_characters(note['Note']))

def merge_ctakes_result_to_csv(icustayids, ctakes_result_path=None, merged_results_path=None, manager_queue=None):
    for icustay in icustayids:
        if manager_queue is not None:
            manager_queue.put(icustay)
        if not os.path.exists(noteevents_path + "{}/".format(icustay)):
            continue
        icustay_path = ctakes_result_path + str(icustay) + '/'
        files = [icustay_path + x for x in os.listdir(icustay_path)]
        data = []
        for file in files:
            record = dict()
            record['timestamp'] = file.split('_')[1]
            with open(file, 'r') as f:
                record['umls'] = f.read()
            data.append(record)
        data = pandas.DataFrame(data)
        data.to_csv(merged_results_path + "{}.csv".format(icustay), index=False)

parameters = functions.load_parameters_file()

dataset_csv = pandas.read_csv(parameters['mimic_data_path'] + parameters['dataset_file_name'])
noteevents_path = parameters['mimic_data_path'] + parameters['noteevents_anonymized_tokens_normalized']
icustays = dataset_csv['icustay_id']
icustays = np.array_split(icustays, 10)
ctakes_data_path = parameters['mimic_data_path'] + parameters['ctakes_data_path']
ctakes_result_data_path = parameters['mimic_data_path'] + parameters['ctakes_output_path']
uids_data_path = parameters['mimic_data_path'] + parameters['noteevents_ctakes_processed_data_path']
if not os.path.exists(ctakes_data_path):
    os.mkdir(ctakes_data_path)
if not os.path.exists(ctakes_result_data_path):
    os.mkdir(ctakes_result_data_path)
if not os.path.exists(uids_data_path):
    os.mkdir(uids_data_path)

with mp.Pool(processes=4) as pool:
    m = mp.Manager()
    queue = m.Queue()
    partial_split_data_ctakes = partial(split_data_for_ctakes,
                                        noteevents_path = noteevents_path,
                                        ctakes_data_path=ctakes_data_path,
                                        manager_queue=queue)
    # TODO : process the fales and put it into a format that can be readable for the ctakes (ID_TIME.txt)
    print("===== Spliting events into different files =====")
    map_obj = pool.map_async(partial_split_data_ctakes, icustays)
    consumed = 0
    while not map_obj.ready():
        for _ in range(queue.qsize()):
            queue.get()
            consumed += 1
        sys.stderr.write('\rdone {0:%}'.format(consumed / len(dataset_csv)))

    # TODO : execute the command line for the ctakes pipeline
    ctakes_params = functions.load_ctakes_parameters_file()
    dirname = os.path.dirname(os.path.realpath(__file__)) + '/'
    ctakes_command = "sh {}bin/runClinicalPipeline.sh  -i {}  --xmiOut {}  --user {}  --pass {}"\
        .format(ctakes_params['ctakes_path'], dirname + ctakes_data_path, dirname + ctakes_result_data_path,
                ctakes_params['umls_username'], ctakes_params['umls_password'])
    print(ctakes_command)
    process = subprocess.Popen(ctakes_command, shell=True, stdout=subprocess.PIPE)
    for line in process.stdout:
        print(line)
    process.wait()
    exit()
    # TODO : merge the files for the same id into a csv with the results for the ctakes
    partial_merge_results = partial(merge_ctakes_result_to_csv, ctakes_result_path=ctakes_result_data_path,
                                    merged_results_path=uids_data_path, manager_queue=queue)
    print("===== Merging events into a csv =====")
    map_obj = pool.map_async(partial_merge_results, icustays)
    consumed = 0
    while not map_obj.ready():
        for _ in range(queue.qsize()):
            queue.get()
            consumed += 1
        sys.stderr.write('\rdone {0:%}'.format(consumed / len(dataset_csv)))