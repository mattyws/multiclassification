"""
This script will fill all missing values.
It's suppose to be used only with numerical features, so make sure to use dataset_nominal_to_hotencoding.py
if that is not the case for the dataset. In the same manner, all the dataset have to have a csv file in the path.
"""
import os
from datetime import timedelta
from functools import partial
from itertools import islice
from math import ceil

import numpy
import pandas as pd
import multiprocessing as mp

import sys

import functions


def fill_missing_values(icustays, events_path=None, new_events_path=None, manager_queue=None):
    for icustay in icustays:
        if os.path.exists(new_events_path+'{}.csv'.format(icustay)):
            continue
        events = pd.read_csv(events_path+'{}.csv'.format(icustay))
        events = events.ffill()
        events = events.bfill()
        events.to_csv(new_events_path+'{}.csv'.format(icustay), index=False)
        if manager_queue is not None:
            manager_queue.put(icustay)




parameters = functions.load_parameters_file()
events_path =  parameters['mimic_data_path'] + "sepsis_all_features_raw_merged/"
new_events_path = parameters['mimic_data_path'] + "sepsis_all_features_no_missing/"
if not os.path.exists(new_events_path):
    os.mkdir(new_events_path)

dataset = pd.read_csv(parameters['mimic_data_path'] + parameters['dataset_file_name'])

total_files = len(dataset)
icustay_ids = list(dataset['icustay_id'])
dataset_for_mp = numpy.array_split(icustay_ids, 10)

m = mp.Manager()
queue = m.Queue()
partial_fill_missing_values = partial(fill_missing_values, events_path=events_path, new_events_path=new_events_path ,
                                      manager_queue=queue)
with mp.Pool(processes=4) as pool:
    map_obj = pool.map_async(partial_fill_missing_values, dataset_for_mp)
    consumed = 0
    while not map_obj.ready():
        for _ in range(queue.qsize()):
            queue.get()
            consumed += 1
        sys.stderr.write('\rdone {0:%}'.format(consumed / total_files))
