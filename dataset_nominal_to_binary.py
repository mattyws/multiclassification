"""
Creates the binary columns for nominal data, adding the columns that doesn't appear at each patients events
"""
import csv
import os
from functools import partial

import pandas as pd
import numpy as np
import multiprocessing as mp
from pandas._libs import json

import helper

parametersFilePath = "parameters.json"

#Loading parameters file
print("========= Loading Parameters")
parameters = None
with open(parametersFilePath, 'r') as parametersFileHandler:
    parameters = json.load(parametersFileHandler)
if parameters is None:
    exit(1)

mimic_data_path = parameters['mimic_data_path']
# TODO: loop through events like is done in dataset_filter_events (only for labevents and chartevents)
features_event_label = 'labevents'
events_files_path = mimic_data_path + 'sepsis_{}/'.format(features_event_label)
new_events_files_path = mimic_data_path + 'sepsis_binary_{}/'.format(features_event_label)
if not os.path.exists(new_events_files_path):
    os.mkdir(new_events_files_path)

dataset_csv = pd.read_csv(parameters['dataset_file_name'])

def binarize_nominal_events(icustay_id, events_files_path, new_events_files_path):
    nominal_events = []
    if os.path.exists(events_files_path + '{}.csv'.format(icustay_id)) and \
            not os.path.exists(new_events_files_path + '{}.csv'.format(icustay_id)):
        print("#### {} ####".format(icustay_id))
        # Get events and change nominal to binary
        events = pd.read_csv(events_files_path + '{}.csv'.format(icustay_id))
        if 'Unnamed: 0' in events.columns:
            events.loc[:, 'Unnamed: 0'] = pd.to_datetime(events['Unnamed: 0'], format=parameters['datetime_pattern'])
            events = events.set_index(['Unnamed: 0'])
            events = events.sort_index()
        events = pd.get_dummies(events, dummy_na=False)
        nominal_events = events.columns
        events.to_csv(new_events_files_path + '{}.csv'.format(icustay_id), quoting=csv.QUOTE_NONNUMERIC)
    elif os.path.exists(new_events_files_path + '{}.csv'.format(icustay_id)):
        print("#### {} #### - Already exists".format(icustay_id))
        events = pd.read_csv(new_events_files_path + '{}.csv'.format(icustay_id))
        if 'Unnamed: 0' in events.columns:
            events = events.drop(columns=['Unnamed: 0'])
        nominal_events = events.columns
    print("#### End {} ####".format(icustay_id))
    return nominal_events

def fill_missing_events(icustay_id, all_features, new_events_files_path):
    print("---- {} ----".format(icustay_id))
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
        events = events.sort_index(axis=1)
        events.to_csv(new_events_files_path + '{}.csv'.format(icustay_id), quoting=csv.QUOTE_NONNUMERIC)
        print("---- End {} ----".format(icustay_id))

# TODO: remove events that are nominal but somehow they still exist


# Using as arg only the icustay_id, bc of fixating the others parameters
args = list(dataset_csv['icustay_id'])
# If the dir already exists and it has files for all dataset already created, only loop to get all possible events
if os.path.exists(new_events_files_path) and len(os.listdir(new_events_files_path)) == len(dataset_csv):
    print("{} already created and have files for all dataset, fetching the columns".format(new_events_files_path))
    file_list = [new_events_files_path + x for x in os.listdir(new_events_files_path)]
    features_after_binarized = set()
    i = 0
    for file in file_list:
        if i % 1000 == 0:
            print("Read {} files".format(i))
        with open(file) as file:
            f = csv.DictReader(file)
            features_after_binarized |= set(f.fieldnames)
            i += 1
else:
    # If doesn't exist, go binarize the values
    # Creating a partial maintaining some arguments with fixed values
    partial_binarize_nominal_events = partial(binarize_nominal_events,
                                              events_files_path=events_files_path,
                                              new_events_files_path=new_events_files_path)
    # The results of the processes
    results = []
    # Creating the pool
    with mp.Pool(processes=6) as pool:
        results = pool.map(partial_binarize_nominal_events, args)
    print("========== Get new features after the dummies ==========")
    features_after_binarized = set()
    for result in results:
        features_after_binarized |= set(result)
    # features_after_binarized = list(features_after_binarized)

# Removing nominal features in raw form if they exists (somehow that happens)
are_nominal = []
for feature in features_after_binarized:
    if len(feature.split('_')) > 2:
        are_nominal.append('_'.join(feature.split('_')[:2]))
print(are_nominal)
exit()

print("========== Filling events ==========")
partial_fill_missing_events = partial(fill_missing_events, all_features=features_after_binarized,
                                      new_events_files_path=new_events_files_path)
with mp.Pool(processes=6) as pool:
    pool.map(partial_fill_missing_events, args)