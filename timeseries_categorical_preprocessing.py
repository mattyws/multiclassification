import json
import os
import textdistance as td
import pandas as pd
from resources import functions
import re

parameters = functions.load_parameters_file()
mimic_data_path = parameters['mimic_data_path']
features_event_label = 'chartevents'
events_files_path = mimic_data_path + 'sepsis_{}2/'.format(features_event_label)

dataset = pd.read_csv(parameters['dataset_file_name'])

# Getting event values if it's not already done
if not os.path.exists('categorical_values.json'):
    categorical_values = dict()
    for index, row in dataset.iterrows():
        print("===== {} =====".format(row['icustay_id']))
        if not os.path.exists(events_files_path + '{}.csv'.format(row['icustay_id'])):
            continue
        events = pd.read_csv(events_files_path + '{}.csv'.format(row['icustay_id']))
        if 'Unnamed: 0' in events.columns:
            events = events.drop(columns=['Unnamed: 0'])
        categorical_events = events.select_dtypes(include=['object'])
        for column in categorical_events.columns:
            if column not in categorical_values.keys():
                categorical_values[column] = set()
            categorical_values[column] |= set(categorical_events[column].dropna().unique())
    for key in categorical_values.keys():
        categorical_values[key] = list(categorical_values[key])
    with open('categorical_values.json', 'w') as file:
        json.dump(categorical_values, file, indent=4)
else:
    categorical_values = json.load(open('categorical_values.json'))

similarity_table = []
for key in categorical_values.keys():
    # preproc_values = [re.sub(r"\s+", "", str(x).lower(), flags=re.UNICODE) for x in categorical_values[key]]
    # preproc_values.sort()
    values = categorical_values[key]
    for value in values:
        preproc_value = re.sub(r"\s+", "", str(value).lower(), flags=re.UNICODE)
        for other in values:
            similarities = dict()
            preproc_other = re.sub(r"\s+", "", str(other).lower(), flags=re.UNICODE)
            similarity = td.levenshtein.normalized_similarity(preproc_value, preproc_other)
            if similarity > 0.8 and similarity != 1:
                similarities["raw string1"] = value
                similarities["raw string2"] = other
                similarities["string1"] = preproc_value
                similarities["string2"] = preproc_other
                similarities["similarity"] = similarity
                similarity_table.append(similarities)
similarity_table = pd.DataFrame(similarity_table)
if len(similarity_table) != 0 :
    similarity_table.to_csv('similaridades.csv')
