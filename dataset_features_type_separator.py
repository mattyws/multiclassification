"""
Separate features with numerical and categorical values
"""
import csv
import os
import pickle
import pprint

import sys
from functools import partial

import functions
import pandas as pd
import numpy as np
import multiprocessing as mp

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 3)

def get_features_type(icustays, patient_events_path=None, manager_queue=None):
    features_types = dict()
    for icustay in icustays:
        if manager_queue is not None:
            manager_queue.put(icustay)
        if not os.path.exists(patient_events_path + "{}.csv".format(icustay)):
            continue
        events = pd.read_csv(patient_events_path + "{}.csv".format(icustay))
        for column in events.columns:
            if column in ['starttime', 'endtime', 'Unnamed: 0', 'chartevents_Unnamed: 0', 'labevents_Unnamed: 0']:
                continue
            aux_column = events[column].copy()
            event_types = set(map(type, aux_column))
            if len(event_types) > 1:
                if str in event_types:
                    aux_column = aux_column.fillna('')
                    event_types = set(map(type, aux_column))
            if column not in features_types.keys():
                features_types[column] = set()
            features_types[column].update(event_types)
    return features_types

def split_features(icustays, features_to_process=None, patient_events_path=None, new_events_path=None,
                   manager_queue=None) :
    for icustay in icustays:
        if manager_queue is not None:
            manager_queue.put(icustay)
        if not os.path.exists(patient_events_path + "{}.csv".format(icustay))\
                or os.path.exists(new_events_path + "{}.csv".format(icustay)):
            continue
        events = pd.read_csv(patient_events_path + "{}.csv".format(icustay))
        features_to_drop = []
        for feature in features_to_process.keys():
            if feature in events.columns:
                aux = events[feature].copy()
                events[feature + '_numeric'] = pd.to_numeric(events[feature], errors='coerce')
                events[feature + '_categorical'] = np.where(events[feature + '_numeric'].isna(), aux, np.nan)
                features_to_drop.append(feature)
                if events[feature + '_numeric'].dropna().empty:
                    features_to_drop.append(feature + '_numeric')
                if events[feature + '_categorical'].dropna().empty:
                    features_to_drop.append(feature + '_categorical')
        events = events.drop(columns=features_to_drop)
        print(icustay, events.columns)
        events = events.sort_index(axis=1)
        events.to_csv(new_events_path + "{}.csv".format(icustay), index=False, quoting=csv.QUOTE_NONNUMERIC)


parameters = functions.load_parameters_file()
pp = pprint.PrettyPrinter(indent=4)

dataset = pd.read_csv(parameters['mimic_data_path'] + parameters['dataset_file_name'])
patient_events_path = parameters['mimic_data_path'] + "sepsis_raw_merged/"
icustays = np.array_split(dataset['icustay_id'], 10)
m = mp.Manager()
queue = m.Queue()

if not os.path.exists(parameters['mimic_data_path'] + parameters['features_types_file_name']):
    print("====== Getting types =====")
    with mp.Pool(processes=6) as pool:
        partial_split_features = partial(get_features_type,
                                         patient_events_path=patient_events_path,
                                         manager_queue=queue)
        map_obj = pool.map_async(partial_split_features, icustays)
        consumed = 0
        while not map_obj.ready():
            for _ in range(queue.qsize()):
                queue.get()
                consumed += 1
            sys.stderr.write('\rdone {0:%}'.format(consumed / len(dataset)))
        results = map_obj.get()
    features_types = {k : v for d in results for k, v in d.items()}
    with open(parameters['mimic_data_path'] + parameters['features_types_file_name'], 'wb') as file:
        pickle.dump(features_types, file, pickle.HIGHEST_PROTOCOL)
else:
    with open(parameters['mimic_data_path'] + parameters['features_types_file_name'], 'rb') as file:
        features_types = pickle.load(file)

features_to_process = dict()
for key in features_types.keys():
    if len(features_types[key]) > 1:
        features_to_process[key] = features_types[key]

i = 0
new_events_path = parameters['mimic_data_path'] + parameters['separated_features_types_dirname']
if not os.path.exists(new_events_path):
    os.mkdir(new_events_path)

print("====== Processing events =====")
with mp.Pool(processes=6) as pool:
    partial_split_features = partial(split_features,
                                     patient_events_path=patient_events_path,
                                     new_events_path=new_events_path,
                                     features_to_process=features_to_process,
                                     manager_queue=queue)
    map_obj = pool.map_async(partial_split_features, icustays)
    consumed = 0
    while not map_obj.ready():
        for _ in range(queue.qsize()):
            queue.get()
            consumed += 1
        sys.stderr.write('\rdone {0:%}'.format(consumed / len(dataset)))