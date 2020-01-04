"""
Remove patients that have only structured data or only textual data.
"""
import os

import pandas as pd
import functions

parameters = functions.load_parameters_file()

dataset = pd.read_csv(parameters['mimic_data_path']+parameters['dataset_file_name'])
print(dataset['class'].value_counts())
noteevents_path = parameters['mimic_data_path'] + 'sepsis_noteevents/'
chartevents_path = parameters['mimic_data_path'] + 'sepsis_raw_merged/'

has_noteevents = set([x.split('.')[0] for x in os.listdir(noteevents_path)])
has_chartevents = set([x.split('.')[0] for x in os.listdir(chartevents_path)])

dont_have_notes = has_chartevents.difference(has_noteevents)
dont_have_chart = has_noteevents.difference(has_chartevents)
dont_have_both = dont_have_chart.union(dont_have_notes)

new_dataset = dataset[~dataset['icustay_id'].isin(list(dont_have_both))]
print(len(dataset), len(new_dataset))

new_dataset.loc[:, 'intime'] = pd.to_datetime(new_dataset['intime'], format=parameters['datetime_pattern'])
new_dataset.loc[:, 'sofa_increasing_time_poe'] = pd.to_datetime(new_dataset['sofa_increasing_time_poe'],
                                                                format=parameters['datetime_pattern'])

new_dataset['intime_diff'] = (new_dataset['sofa_increasing_time_poe'] - new_dataset['intime'])\
    .apply(lambda x : x.seconds//3600)

print(len(new_dataset))
print(new_dataset['class'].value_counts())
filtered_dataset = new_dataset[new_dataset['intime_diff'] < 7]
filtered_dataset = new_dataset[~new_dataset['icustay_id'].isin(filtered_dataset['icustay_id'])]
print(len(filtered_dataset))
print(filtered_dataset['class'].value_counts())


filtered_dataset.to_csv(parameters['mimic_data_path']+'new_'+parameters['dataset_file_name'])