"""
Separate features with numerical and categorical values
"""
import os
import pickle
import pprint

import sys

import functions
import pandas as pd
import numpy as np

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 3)

parameters = functions.load_parameters_file()
pp = pprint.PrettyPrinter(indent=4)

dataset = pd.read_csv(parameters['mimic_data_path'] + parameters['insight_dataset_file_name'])
patient_events_path = parameters['mimic_data_path'] + "sepsis_raw_merged/"

if not os.path.exists(parameters['mimic_data_path'] + parameters['features_types_file_name']):
    features_types = dict()
    values_for_mixed_in_patient = dict()
    i = 0
    for icustay in dataset['icustay_id']:
        sys.stderr.write('\rdone {0:%}'.format(i / len(dataset)))
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
                    # Have mixed types inside this patient
                    if len(event_types) > 1:
                        if column not in values_for_mixed_in_patient.keys():
                            values_for_mixed_in_patient[column] = []
            if column not in features_types.keys():
                features_types[column] = set()
            features_types[column].update(event_types)
            if column in values_for_mixed_in_patient.keys():
                values_for_mixed_in_patient[column].extend(aux_column.value_counts().index.tolist())
            # print(events[column].dtype)
            # exit()
            # if column not in features_types.keys():
            #     features_types[column] = set()
            # features_types[column].add(events[column].dtype)
        i += 1
    with open(parameters['mimic_data_path'] + parameters['features_types_file_name'], 'wb') as file:
        pickle.dump(features_types, file, pickle.HIGHEST_PROTOCOL)
    with open(parameters['mimic_data_path'] + parameters['values_for_mixed_in_patient_file_name'], 'wb') as file:
        pickle.dump(values_for_mixed_in_patient, file, pickle.HIGHEST_PROTOCOL)
else:
    with open(parameters['mimic_data_path'] + parameters['features_types_file_name'], 'rb') as file:
        features_types = pickle.load(file)
    with open(parameters['mimic_data_path'] + parameters['values_for_mixed_in_patient_file_name'], 'rb') as file:
        values_for_mixed_in_patient = pickle.load(file)

features_to_process = dict()
for key in features_types.keys():
    if len(features_types[key]) > 1:
        features_to_process[key] = features_types[key]

i = 0
new_events_path = parameters['mimic_data_path'] + parameters['separated_features_types_dirname']
if not os.path.exists(new_events_path):
    os.mkdir(new_events_path)
for icustay in dataset['icustay_id']:
    sys.stderr.write('\rdone {0:%}'.format(i / len(dataset)))
    if not os.path.exists(patient_events_path + "{}.csv".format(icustay)):
        continue
    events = pd.read_csv(patient_events_path + "{}.csv".format(icustay))
    features_to_drop = []
    for feature in features_to_process.keys():
        if feature in events.columns:
            aux = events[feature].copy()
            events[feature + '_numeric'] = pd.to_numeric(events[feature], errors='coerce')
            events[feature + '_categorical'] = np.where(events[feature+'_numeric'].isna(), aux, np.nan)
            features_to_drop.append(feature)
            if events[feature + '_numeric'].dropna().empty:
                features_to_drop.append(feature + '_numeric')
            if events[feature + '_categorical'].dropna().empty:
                features_to_drop.append(feature + '_categorical')
    events = events.drop(columns=features_to_drop)
    events = events.sort_index(axis=1)
    events.to_csv(new_events_path + "{}.csv".format(icustay))
    if i == 10:
        exit()
    i+= 1