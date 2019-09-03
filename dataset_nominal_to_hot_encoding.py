"""
Creates the binary columns for nominal data, adding the columns that doesn't appear at each patients events
# TODO: save features names in a file - for each patient/ or the final feature set
"""
import csv
import os
from functools import partial
from itertools import product

import numpy
import pandas as pd
import numpy as np
import multiprocessing as mp

import sys
from pandas._libs import json


def hot_encoding_nominal_features(icustay_ids, feature_type, events_files_path="", new_events_files_path="", manager_queue=None):
    nominal_events = set()
    events_files_path = events_files_path.format(feature_type)
    new_events_files_path = new_events_files_path.format(feature_type)
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
            nominal_events = set(events.columns)
            events.to_csv(new_events_files_path + '{}.csv'.format(icustay_id), quoting=csv.QUOTE_NONNUMERIC)
        elif os.path.exists(new_events_files_path + '{}.csv'.format(icustay_id)):
            events = pd.read_csv(new_events_files_path + '{}.csv'.format(icustay_id))
            if 'Unnamed: 0' in events.columns:
                events = events.drop(columns=['Unnamed: 0'])
            nominal_events = set(events.columns)
        if manager_queue is not None:
            manager_queue.put(feature_type, nominal_events)
    # return feature_type, nominal_events

def create_missing_features(icustay_ids, feature_type, all_features, new_events_files_path, are_nominal, manager_queue=None):
    new_events_files_path = new_events_files_path.format(feature_type)
    for icustay_id in icustay_ids:
        if os.path.exists(new_events_files_path + '{}.csv'.format(icustay_id)):
            events = pd.read_csv(new_events_files_path + '{}.csv'.format(icustay_id))
            if 'Unnamed: 0' in events.columns:
                events = events.set_index(['Unnamed: 0'])
            # events = events.fillna(0)
            if len(events.columns) != len(all_features[feature_type]):
                # zeros = np.zeros(len(events))
                features_not_in = all_features[feature_type] - set(events.columns)
                for itemid in features_not_in:
                    events[itemid] = np.nan
            events = events.drop(columns=are_nominal[feature_type], errors='ignore')
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
features_types = ['raw_merged']
for feature_type in features_types:
    new_events_files_path = mimic_data_path + parameters["hotencoded_events_dirname"].format(feature_type)
    if not os.path.exists(new_events_files_path):
        os.mkdir(new_events_files_path)

dataset_csv = pd.read_csv(parameters["mimic_data_path"] + parameters['dataset_file_name'])
# Using as arg only the icustay_id, bc of fixating the others parameters
total_files = len(dataset_csv) * len(features_types)
results = []
with mp.Pool(processes=4) as pool:
    manager = mp.Manager()
    queue = manager.Queue()
    # Generate data to be used by the processes
    args = numpy.array_split(dataset_csv['icustay_id'], 10)
    args = product(args, features_types)
    # Creating a partial maintaining some arguments with fixed values
    partial_hot_encoding_nominal_features = partial(hot_encoding_nominal_features,
                                                    events_files_path=mimic_data_path + parameters["raw_events_dirname"],
                                                    new_events_files_path=mimic_data_path + parameters["hotencoded_events_dirname"],
                                                    manager_queue=queue)

    print("===== Hot encoding nominal features =====")
    consumed = 0
    # Using starmap to pass the tuple as separated parameters to the functions
    map_obj = pool.starmap_async(partial_hot_encoding_nominal_features, args)
    features_after_binarized = dict()
    while not map_obj.ready():
        for _ in range(queue.qsize()):
            result = queue.get()
            if result[0] not in features_after_binarized.keys():
                features_after_binarized[result[0]] = set()
            features_after_binarized[result[0]].update(result[1])
            consumed += 1
        sys.stderr.write('\rdone {0:%}'.format(consumed / total_files))
    # results = map_obj.get()

    # print("===== Hot encoding nominal features =====")
    # consumed = 0
    # # Using starmap to pass the tuple as separated parameters to the functions
    # map_obj = pool.starmap_async(partial_hot_encoding_nominal_features, args)
    # while not map_obj.ready():
    #     for _ in range(queue.qsize()):
    #         queue.get()
    #         consumed += 1
    #     sys.stderr.write('\rdone {0:%}'.format(consumed / total_files))
    # results = map_obj.get()
    #
    # print("====== Joining features to get all features =====")
    # consumed = 0
    # features_after_binarized = dict()
    # total_results = len(results)
    # for result in results:
    #     if result[0] not in features_after_binarized.keys():
    #         features_after_binarized[result[0]] = set()
    #     features_after_binarized[result[0]].update(result[1])
    #     sys.stderr.write('\rdone {0:%}'.format(consumed / total_results))
    #     consumed += 1

    # Removing nominal features in raw form if they exists (somehow that happens)
    print("====== Removing hot encoded raw features from set of all features ======")
    are_nominal = dict()
    for key in features_after_binarized.keys():
        if key not in are_nominal:
            are_nominal[key] = set()
        for feature in features_after_binarized[key]:
            if len(feature.split('_')) >= 2:
                are_nominal[key].add(feature.split('_')[0])
    print(are_nominal[key])
    for key in are_nominal.keys():
        for feature in are_nominal[key]:
            features_after_binarized[key].discard(feature)

    # Re-generating data as it was already iterated
    args = numpy.array_split(dataset_csv['icustay_id'], 10)
    args = product(args, features_types)
    partial_create_missing_features = partial(create_missing_features, all_features=features_after_binarized,
                                              new_events_files_path=mimic_data_path + parameters["hotencoded_events_dirname"],
                                              are_nominal=are_nominal,
                                              manager_queue=queue)
    print("===== Filling missing features =====")
    map_obj = pool.starmap_async(partial_create_missing_features, args)
    consumed = 0
    while not map_obj.ready():
        for _ in range(queue.qsize()):
            queue.get()
            consumed += 1
        sys.stderr.write('\rdone {0:%}'.format(consumed / total_files))