"""
This script will merge all events that occur at the same hour (taking patient's intime at the icu as the start),
into one events, using the mean of the values at this window.
This script will also remove any patient file that doesn't have any events.
"""
import os
from datetime import timedelta
from functools import partial
from itertools import islice
from math import ceil

import numpy
import pandas as pd
import multiprocessing as mp

import sys

import functions


def process_events(dataset, events_path, new_events_path, datetime_pattern='%Y-%m-%d %H:%M:%S',
                   manager_queue=None):
    icustays_to_remove = []
    for index, patient in dataset.iterrows():
        if not os.path.exists(events_path+'{}.csv'.format(patient['icustay_id'])):
            continue
        events = pd.read_csv(events_path+'{}.csv'.format(patient['icustay_id']))
        # Unnamed: 0 here represents the time for the events
        events.loc[:, 'Unnamed: 0'] = pd.to_datetime(events['Unnamed: 0'], format=datetime_pattern)
        # events.loc[: 'starttime']
        starttime = patient['intime'].replace(minute=0, second=0)
        buckets = []
        if pd.isna(patient['sofa_increasing_time_poe']):
            cut_time = patient['outtime']
        else:
            cut_time = patient['sofa_increasing_time_poe']
        while starttime < cut_time:
            endtime = starttime + timedelta(hours=1)
            bucket = dict()
            bucket['starttime'] = starttime
            bucket['endtime'] = endtime
            bucket_events = events[(events['Unnamed: 0'] > starttime) & (events['Unnamed: 0'] <= endtime)]
            for column in bucket_events.columns:
                if column == 'Unnamed: 0':
                    continue
                # if '_categorical' in column \
                #     or '_categorical' not in column and '_numeric' not in column and len(column.split('_')) >= 3:
                #     if 1 in bucket_events[column].tolist():
                #         bucket[column] = 1
                #     else:
                #         bucket[column] = 0
                # else:
                values = bucket_events[column].dropna().tolist()
                if len(values) != 0:
                    bucket[column] = sum(values)/len(values)
                    bucket[column + "_min"] = min(values)
                    bucket[column + "_max"] = max(values)
                else:
                    bucket[column] = numpy.nan
                    bucket[column + "_min"] = numpy.nan
                    bucket[column + "_max"] = numpy.nan
            buckets.append(bucket)
            starttime += timedelta(hours=1)
        buckets = pd.DataFrame(buckets)
        buckets = buckets.sort_index(axis=1)
        if buckets.empty:
            icustays_to_remove.append(patient['icustay_id'])
        else:
            buckets.to_csv(new_events_path+'{}.csv'.format(patient['icustay_id']), index=False)
        if manager_queue is not None:
            manager_queue.put(index)
    return icustays_to_remove


parameters = functions.load_parameters_file()
events_path =  parameters['mimic_data_path'] + "sepsis_articles/"
new_events_path = parameters['mimic_data_path'] + "sepsis_articles_bucket/"
if not os.path.exists(new_events_path):
    os.mkdir(new_events_path)

dataset = pd.read_csv(parameters['mimic_data_path'] + parameters['dataset_file_name'])
if 'Unnamed: 0' in dataset.columns:
    dataset = dataset.drop(columns=['Unnamed: 0'])
dataset.loc[:, 'intime'] = pd.to_datetime(dataset['intime'], format=parameters['datetime_pattern'])
dataset.loc[:, 'outtime'] = pd.to_datetime(dataset['outtime'], format=parameters['datetime_pattern'])
dataset.loc[:, 'sofa_increasing_time_poe'] = pd.to_datetime(dataset['sofa_increasing_time_poe'],
                                                             format=parameters['datetime_pattern'])
original_len = len(dataset)
total_files = len(dataset)
icustay_ids = list(dataset['icustay_id'])
dataset_for_mp = numpy.array_split(dataset, 10)


with mp.Pool(processes=6) as pool:
    m = mp.Manager()
    queue = m.Queue()
    partial_normalize_files = partial(process_events, events_path=events_path, new_events_path=new_events_path,
                                  manager_queue=queue)
    map_obj = pool.map_async(partial_normalize_files, dataset_for_mp)
    consumed = 0
    while not map_obj.ready():
        for _ in range(queue.qsize()):
            queue.get()
            consumed += 1
        sys.stderr.write('\rdone {0:%}'.format(consumed / total_files))
    result = map_obj.get()
    result = numpy.concatenate(result, axis=0)
    dataset_to_remove = dataset[dataset['icustay_id'].isin(result)]
    dataset = dataset.drop(dataset_to_remove.index)
    print(original_len, len(dataset))
    if len(dataset) != original_len:
        print("Removed some data")
        dataset.to_csv(parameters['mimic_data_path'] + parameters['dataset_file_name'] + '2')
