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


def fill_missing_values(dataset, events_path, manager_queue=None):
    for index, patient in dataset.iterrows():
        events = pd.read_csv(events_path+'{}.csv'.format(patient['icustay_id']))
        events = events.fillna(0)
        events.to_csv(events_path+'{}.csv'.format(patient['icustay_id']), index=False)
        if manager_queue is not None:
            manager_queue.put(index)




parameters = functions.load_parameters_file()
events_path =  parameters['mimic_data_path'] + "sepsis_insight_bucket/"

dataset = pd.read_csv(parameters['mimic_data_path'] + parameters['dataset_file_name'])

total_files = len(dataset)
icustay_ids = list(dataset['icustay_id'])
dataset_for_mp = numpy.array_split(dataset, 10)

m = mp.Manager()
queue = m.Queue()
partial_fill_missing_values = partial(fill_missing_values, events_path=events_path, manager_queue=queue)
with mp.Pool(processes=len(dataset_for_mp)) as pool:
    map_obj = pool.map_async(partial_fill_missing_values, dataset_for_mp)
    consumed = 0
    while not map_obj.ready():
        for _ in range(queue.qsize()):
            queue.get()
            consumed += 1
        sys.stderr.write('\rdone {0:%}'.format(consumed / total_files))
