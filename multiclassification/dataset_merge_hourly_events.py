"""
This script will merge all events that occur at the same hour (taking patient's intime at the icu as the start),
into one events, using the mean of the values at this window.
This script will also remove any patient file that doesn't have any events.
"""
import multiprocessing as mp
import os
import sys
from datetime import timedelta
from functools import partial

import numpy
import pandas as pd


def process_events(dataset, events_path, new_events_path, datetime_pattern='%Y-%m-%d %H:%M:%S',
                   manager_queue=None):
    icustays_to_remove = []
    for index, patient in dataset.iterrows():
        if not os.path.exists(events_path+'{}.csv'.format(patient['ICUSTAY_ID'])) or \
                    os.path.exists(new_events_path+'{}.csv'.format(patient['ICUSTAY_ID'])):
            continue
        events = pd.read_csv(events_path+'{}.csv'.format(patient['ICUSTAY_ID']))
        # Unnamed: 0 here represents the time for the events
        events.loc[:, 'Unnamed: 0'] = pd.to_datetime(events['Unnamed: 0'], format=datetime_pattern)
        # events.loc[: 'starttime']
        starttime = patient['INTIME'].replace(minute=0, second=0)
        buckets = []
        cut_time = patient['OUTTIME']
        bucket_num = 0
        while starttime < cut_time:
            endtime = starttime + timedelta(hours=1)
            bucket = dict()
            bucket['starttime'] = starttime
            bucket['endtime'] = endtime
            bucket['bucket'] = bucket_num
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
            bucket_num += 1
        buckets = pd.DataFrame(buckets)
        buckets = buckets.sort_values(by=['starttime'])
        if buckets.empty:
            icustays_to_remove.append(patient['ICUSTAY_ID'])
        else:
            buckets.to_csv(new_events_path+'{}.csv'.format(patient['ICUSTAY_ID']))
        if manager_queue is not None:
            manager_queue.put(index)
    return icustays_to_remove


from multiclassification.parameters.dataset_parameters import parameters

events_path =  parameters['mimic_data_path'] + parameters['multiclassification_directory'] \
               + parameters['features_filtered_dirname']
new_events_path = parameters['mimic_data_path'] + parameters['multiclassification_directory'] \
                  + parameters['events_hourly_merged_dirname']
if not os.path.exists(new_events_path):
    os.mkdir(new_events_path)

dataset = pd.read_csv(parameters['mimic_data_path'] + parameters['multiclassification_directory']
                      + parameters['all_stays_csv_w_events'])
dataset.loc[:, 'INTIME'] = pd.to_datetime(dataset['INTIME'], format=parameters['datetime_pattern'])
dataset.loc[:, 'OUTTIME'] = pd.to_datetime(dataset['OUTTIME'], format=parameters['datetime_pattern'])

original_len = len(dataset)
total_files = len(dataset)
icustay_ids = list(dataset['ICUSTAY_ID'])
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
