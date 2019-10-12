"""
Remove features that have a low frequency of occurrence in patients.
"""
import csv
import os
import pickle
import pprint

import sys
from functools import partial
from math import floor

import functions
import pandas as pd
import numpy as np
import multiprocessing as mp

def get_features_frequency(icustays, patient_events_path=None, manager_queue=None):
    features_frequency = dict()
    for icustay in icustays:
        if manager_queue is not None:
            manager_queue.put(icustay)
        if not os.path.exists(patient_events_path + "{}.csv".format(icustay)):
            continue
        events = pd.read_csv(patient_events_path + "{}.csv".format(icustay))
        for feature in events.columns:
            if feature not in features_frequency.keys():
                features_frequency[feature] = 0
            features_frequency[feature] += 1
    return features_frequency

def remove_low_frequency_features(icustays, patient_events_path=None, new_events_path = None,
                                  features_to_remove=None, manager_queue=None, features_to_remove_from_patient=None):
    for icustay in icustays:
        if manager_queue is not None:
            manager_queue.put(icustay)
        if not os.path.exists(patient_events_path + "{}.csv".format(icustay))\
                or os.path.exists(new_events_path + "{}.csv".format(icustay)):
            continue
        events = pd.read_csv(patient_events_path + "{}.csv".format(icustay))
        if features_to_remove_from_patient is None:
            features_to_remove_from_patient = list()
            for column in events.columns:
                if 'numerical' in column or 'categorical' in column:
                    column_feature = '_'.join(column.split('_')[:3])
                else:
                    column_feature = '_'.join(column.split('_')[:2])
                for feature in features_to_remove:
                    if feature == column_feature:
                        features_to_remove_from_patient.append(column)
            return features_to_remove_from_patient
        # features_in_patient = set(events.columns)
        # features_to_remove_from_patient = list(features_to_remove.intersection(features_in_patient))
        events = events.drop(columns=features_to_remove_from_patient)
        events = events.sort_index(axis=1)
        events.to_csv(new_events_path + "{}.csv".format(icustay), index=False, quoting=csv.QUOTE_NONNUMERIC)

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 3)

parameters = functions.load_parameters_file()
pp = pprint.PrettyPrinter(indent=4)

dataset = pd.read_csv(parameters['mimic_data_path'] + parameters['dataset_file_name'])
patient_events_path = parameters['mimic_data_path'] + "sepsis_raw_merged/"
icustays = np.array_split(dataset['icustay_id'], 10)
m = mp.Manager()
queue = m.Queue()


def merge_results(results):
    final_dict = dict()
    for result in results:
        for key in result:
            if key not in final_dict.keys():
                final_dict[key] = 0
            final_dict[key] += result[key]
    return final_dict


if not os.path.exists(parameters['mimic_data_path'] + parameters['features_frequency_file_name']):
    print("====== Getting frequencies =====")
    with mp.Pool(processes=6) as pool:
        partial_remove_low_frequency_features = partial(get_features_frequency,
                                                        patient_events_path=patient_events_path,
                                                        manager_queue=queue)
        map_obj = pool.map_async(partial_remove_low_frequency_features, icustays)
        consumed = 0
        while not map_obj.ready():
            for _ in range(queue.qsize()):
                queue.get()
                consumed += 1
            sys.stderr.write('\rdone {0:%}'.format(consumed / len(dataset)))
        results = map_obj.get()
    features_frequency = merge_results(results)


    with open(parameters['mimic_data_path'] + parameters['features_frequency_file_name'], 'wb') as file:
        pickle.dump(features_frequency, file)
else:
    with open(parameters['mimic_data_path'] + parameters['features_frequency_file_name'], 'rb') as file:
        features_frequency = pickle.load(file)

index = lambda x, l: floor((x/l) * 10)
frequency_bins = dict()
features_to_remove = set()
for feature in features_frequency.keys():
    bin = index(features_frequency[feature], len(dataset))
    if bin not in frequency_bins.keys():
        frequency_bins[bin] = 0
    frequency_bins[bin] += 1
    if bin == 0:
        features_to_remove.add(feature)
pp.pprint(frequency_bins)
features_to_remove = list(features_to_remove)
features_to_remove.sort()
i = 0
new_events_path = parameters['mimic_data_path'] + parameters['features_low_frequency_removed_dirname']
if not os.path.exists(new_events_path):
    os.mkdir(new_events_path)

print("====== Removing low frequency features =====")
with mp.Pool(processes=6) as pool:
    partial_remove_low_frequency_features = partial(remove_low_frequency_features,
                                                    patient_events_path=parameters['mimic_data_path']
                                                                        + "sepsis_all_features_no_missing/",
                                                    new_events_path=new_events_path,
                                                    features_to_remove=features_to_remove,
                                                    manager_queue=queue)
    features_to_remove_from_patient = partial_remove_low_frequency_features(icustays[0])
    partial_remove_low_frequency_features = partial(partial_remove_low_frequency_features,
                                                    features_to_remove_from_patient=features_to_remove_from_patient)
    map_obj = pool.map_async(partial_remove_low_frequency_features, icustays)
    consumed = 0
    while not map_obj.ready():
        for _ in range(queue.qsize()):
            queue.get()
            consumed += 1
        sys.stderr.write('\rdone {0:%}'.format(consumed / len(dataset)))