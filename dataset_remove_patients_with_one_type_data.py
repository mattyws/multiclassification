"""
Remove patients that have only structured data or only textual data.
"""
import os

import pandas as pd
import functions

parameters = functions.load_parameters_file()

dataset = pd.read_csv(parameters['mimic_data_path']+parameters['dataset_file_name'])
noteevents_path = parameters['mimic_data_path'] + 'sepsis_noteevents/'
chartevents_path = parameters['mimic_data_path'] + 'sepsis_raw_merged/'

has_noteevents = set([x.split('.')[0] for x in os.listdir(noteevents_path)])
has_chartevents = set([x.split('.')[0] for x in os.listdir(chartevents_path)])

dont_have_notes = has_chartevents.difference(has_noteevents)
dont_have_chart = has_noteevents.difference(has_chartevents)
dont_have_both = dont_have_chart.union(dont_have_notes)

new_dataset = dataset[~dataset['icustay_id'].isin(list(dont_have_both))]
print(len(dataset), len(new_dataset))

new_dataset.to_csv(parameters['mimic_data_path']+'new_'+parameters['dataset_file_name'])