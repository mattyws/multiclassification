"""
Creates the binary columns for nominal data, adding the columns that doesn't appear at each patients events
"""
import csv
import os
from functools import partial

import numpy
import pandas as pd
import numpy as np
import multiprocessing as mp

import sys
from pandas._libs import json


def hot_encoding_nominal_features(icustay_ids, events_files_path, new_events_files_path, manager_queue=None):
    nominal_events = set()
    for icustay_id in icustay_ids:
        if os.path.exists(events_files_path + '{}.csv'.format(icustay_id)) and \
                not os.path.exists(new_events_files_path + '{}.csv'.format(icustay_id)):
            # Get events and change nominal to binary
            events = pd.read_csv(events_files_path + '{}.csv'.format(icustay_id))
            if 'Unnamed: 0' in events.columns:
                events.loc[:, 'Unnamed: 0'] = pd.to_datetime(events['Unnamed: 0'], format=parameters['datetime_pattern'])
                events = events.set_index(['Unnamed: 0'])
                events = events.sort_index()
            events = pd.get_dummies(events, dummy_na=False)
            nominal_events.update(events.columns)
            events.to_csv(new_events_files_path + '{}.csv'.format(icustay_id), quoting=csv.QUOTE_NONNUMERIC)
        elif os.path.exists(new_events_files_path + '{}.csv'.format(icustay_id)):
            events = pd.read_csv(new_events_files_path + '{}.csv'.format(icustay_id))
            if 'Unnamed: 0' in events.columns:
                events = events.drop(columns=['Unnamed: 0'])
            nominal_events.update(events.columns)
        if manager_queue is not None:
            manager_queue.put(icustay_id)
    return nominal_events

def create_missing_features(icustay_ids, all_features, new_events_files_path, are_nominal, manager_queue=None):
    for icustay_id in icustay_ids:
        if os.path.exists(new_events_files_path + '{}.csv'.format(icustay_id)):
            events = pd.read_csv(new_events_files_path + '{}.csv'.format(icustay_id))
            if 'Unnamed: 0' in events.columns:
                events = events.set_index(['Unnamed: 0'])
            # events = events.fillna(0)
            if len(events.columns) != len(all_features):
                # zeros = np.zeros(len(events))
                features_not_in = all_features - set(events.columns)
                for itemid in features_not_in:
                    events[itemid] = np.nan
            events = events.drop(columns=are_nominal, errors='ignore')
            events = events.sort_index(axis=1)
            events.to_csv(new_events_files_path + '{}.csv'.format(icustay_id), quoting=csv.QUOTE_NONNUMERIC)
        if manager_queue is not None:
            manager_queue.put(icustay_id)

parametersFilePath = "parameters.json"

# Loading parameters file
print("====== Loading Parameters =====")
parameters = None
with open(parametersFilePath, 'r') as parametersFileHandler:
    parameters = json.load(parametersFileHandler)
if parameters is None:
    exit(1)

mimic_data_path = parameters['mimic_data_path']
features_event_label = 'chartevents'
events_files_path = mimic_data_path + 'sepsis_{}/'.format(features_event_label)
new_events_files_path = mimic_data_path + 'sepsis_binary_{}/'.format(features_event_label)
if not os.path.exists(new_events_files_path):
    os.mkdir(new_events_files_path)

dataset_csv = pd.read_csv(parameters['dataset_file_name'])
# Using as arg only the icustay_id, bc of fixating the others parameters
args = numpy.array_split(dataset_csv['icustay_id'], 10)
results = []
total_files = len(dataset_csv)
# Creating the pool
with mp.Pool(processes=6) as pool:
    manager = mp.Manager()
    queue = manager.Queue()
    # Creating a partial maintaining some arguments with fixed values
    partial_hot_encoding_nominal_features = partial(hot_encoding_nominal_features,
                                                    events_files_path=events_files_path,
                                                    new_events_files_path=new_events_files_path,
                                                    manager_queue=queue)
    map_obj = pool.map_async(partial_hot_encoding_nominal_features, args)
    consumed = 0
    print("===== Hot encoding nominal features =====")
    while not map_obj.ready():
        for _ in range(queue.qsize()):
            queue.get()
            consumed += 1
        sys.stderr.write('\rdone {0:%}'.format(consumed / total_files))
    results = map_obj.get()
    # results = pool.map(partial_binarize_nominal_events, args)
print("====== Joining features to get all features =====")
features_after_binarized = set()
consumed = 0
total_results = len(results)
for result in results:
    features_after_binarized.update(result)
    sys.stderr.write('\rdone {0:%}'.format(consumed / total_results))

# Removing nominal features in raw form if they exists (somehow that happens)
print("====== Removing binary raw features from set of all features ======")
are_nominal = set()
for feature in features_after_binarized:
    if len(feature.split('_')) > 2:
        are_nominal.add('_'.join(feature.split('_')[:2]))
for feature in are_nominal:
    features_after_binarized.discard(feature)

with mp.Pool(processes=6) as pool:
    manager = mp.Manager()
    queue = manager.Queue()
    partial_create_missing_features = partial(create_missing_features, all_features=features_after_binarized,
                                              new_events_files_path=new_events_files_path, are_nominal=are_nominal,
                                              manager_queue=queue)
    map_obj = pool.map_async(partial_create_missing_features, args)
    consumed = 0
    print("===== Filling missing features =====")
    while not map_obj.ready():
        for _ in range(queue.qsize()):
            queue.get()
            consumed += 1
        sys.stderr.write('\rdone {0:%}'.format(consumed / total_files))