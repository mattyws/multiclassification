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

parametersFilePath = "parameters.json"

#Loading parameters file
print("========= Loading Parameters")
parameters = None
with open(parametersFilePath, 'r') as parametersFileHandler:
    parameters = json.load(parametersFileHandler)
if parameters is None:
    exit(1)

mimic_data_path = parameters['mimic_data_path']
new_events_files_path = mimic_data_path + 'sepsis_raw_merged/'
if not os.path.exists(new_events_files_path):
    os.mkdir(new_events_files_path)
chartevents_files_path= mimic_data_path + "sepsis_chartevents/"
labevents_files_path= mimic_data_path + "sepsis_labevents/"

dataset_csv = pd.read_csv(parameters['dataset_file_name'])

def merge_events(icustay_id, chartevents_files_path, labevents_files_path, new_events_files_path, datetime_pattern):
    if not os.path.exists(new_events_files_path+'{}.csv'.format(icustay_id)) :
        print("#### {} ####".format(icustay_id))
        if os.path.exists(chartevents_files_path + '{}.csv'.format(icustay_id)):
            chartevents = pd.read_csv(chartevents_files_path + '{}.csv'.format(icustay_id))
            chartevents.loc[:, 'Unnamed: 0'] = pd.to_datetime(chartevents['Unnamed: 0'], format=datetime_pattern)
            chartevents = chartevents.set_index(["Unnamed: 0"])
        else:
            chartevents = None
        if os.path.exists(labevents_files_path + '{}.csv'.format(icustay_id)):
            labevents = pd.read_csv(labevents_files_path + '{}.csv'.format(icustay_id))
            labevents.loc[:, 'Unnamed: 0'] = pd.to_datetime(labevents['Unnamed: 0'], format=datetime_pattern)
            labevents = labevents.set_index(["Unnamed: 0"])
        else:
            labevents = None
        if chartevents is None:
            chartevents = pd.DataFrame([])
        if labevents is None:
            labevents = pd.DataFrame([])
        chartevents = chartevents.add_prefix('chartevents_')
        labevents = labevents.add_prefix('labevents_')
        events = pd.merge(chartevents, labevents, how='outer', left_index=True, right_index=True)
        if 'labevents_Unnamed: 0' in events.columns:
            events = events.drop(columns=['labevents_Unnamed: 0'])
        if 'chartevents_Unnamed: 0' in events.columns:
            events = events.drop(columns=['chartevents_Unnamed: 0'])
        events.to_csv(new_events_files_path + '{}.csv'.format(icustay_id), quoting=csv.QUOTE_NONNUMERIC)

# Using as arg only the icustay_id, bc of fixating the others parameters
args = list(dataset_csv['icustay_id'])
# Creating a partial maintaining some arguments with fixed values
partial_merge_events = partial(merge_events,
                               chartevents_files_path=chartevents_files_path,
                               labevents_files_path=labevents_files_path,
                               new_events_files_path=new_events_files_path,
                               datetime_pattern = parameters['datetime_pattern'])
# The results of the processes
results = []
# Creating the pool
with mp.Pool(processes=6) as pool:
    pool.map(partial_merge_events, args)
