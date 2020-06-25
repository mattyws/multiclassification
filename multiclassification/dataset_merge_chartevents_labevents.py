"""
Creates the binary columns for nominal data, adding the columns that doesn't appear at each patients events
"""
import csv
import multiprocessing as mp
import os
import sys
from functools import partial

import numpy
import pandas as pd

parametersFilePath = "parameters.json"

#Loading parameters file
print("========= Loading Parameters")
from multiclassification.parameters.dataset_parameters import parameters

mimic_data_path = parameters['mimic_data_path']
new_events_files_path = mimic_data_path + parameters['multiclassification_directory'] \
                        + parameters['merged_events_dirname']
if not os.path.exists(new_events_files_path):
    os.mkdir(new_events_files_path)
chartevents_files_path= mimic_data_path + parameters['multiclassification_directory'] \
                        + parameters['raw_events_dirname'].format('chartevents')
labevents_files_path= mimic_data_path + parameters['multiclassification_directory'] \
                      + parameters['raw_events_dirname'].format('labevents')

dataset_csv = pd.read_csv(parameters['mimic_data_path'] + parameters['multiclassification_directory']
                          + parameters['all_stays_csv'])

def merge_events(icustay_ids, chartevents_files_path, labevents_files_path, new_events_files_path, datetime_pattern,
                 manager_queue=None):
    for icustay_id in icustay_ids:
        if not os.path.exists(new_events_files_path+'{}.csv'.format(icustay_id)) :
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
            if chartevents is None and labevents is None:
                continue
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
            if manager_queue is not None:
                manager_queue.put(icustay_id)

total_files = len(dataset_csv)
# Using as arg only the icustay_id, bc of fixating the others parameters
args = numpy.array_split(list(dataset_csv['ICUSTAY_ID']), 10)
# Creating a partial maintaining some arguments with fixed values
with mp.Pool(processes=6) as pool:
    m = mp.Manager()
    queue = m.Queue()
    partial_merge_events = partial(merge_events,
                                   chartevents_files_path=chartevents_files_path,
                                   labevents_files_path=labevents_files_path,
                                   new_events_files_path=new_events_files_path,
                                   datetime_pattern=parameters['datetime_pattern'],
                                   manager_queue=queue)
    map_obj = pool.map_async(partial_merge_events, args)
    consumed = 0
    while not map_obj.ready():
        for _ in range(queue.qsize()):
            queue.get()
            consumed += 1
        sys.stderr.write('\rdone {0:%}'.format(consumed / total_files))