"""
This script will concatenate all noteevents that occur at the same hour (taking patient's intime at the icu as the start),
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
        events.loc[:, 'charttime'] = pd.to_datetime(events['charttime'], format=datetime_pattern)
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
            bucket_events = events[(events['charttime'] > starttime) & (events['charttime'] <= endtime)].dropna()
            if not bucket_events.empty:
                text = bucket_events['preprocessed_note'].str.cat(sep=" ")
                if not pd.isna(text) and not len(text.strip()) == 0:
                    bucket['text'] = text
                    buckets.append(bucket)
            starttime += timedelta(hours=1)
            bucket_num += 1
        buckets = pd.DataFrame(buckets)
        buckets = buckets.sort_values(by=['starttime'])
        buckets = buckets.dropna()
        if buckets.empty:
            icustays_to_remove.append(patient['ICUSTAY_ID'])
        else:
            buckets.to_csv(new_events_path+'{}.csv'.format(patient['ICUSTAY_ID']))
        if manager_queue is not None:
            manager_queue.put(index)
    return icustays_to_remove


from multiclassification.parameters.dataset_parameters import parameters

events_path =  parameters['mimic_data_path'] + parameters['multiclassification_directory'] \
               + parameters['noteevents_anonymized_tokens_normalized_preprocessed']
new_events_path = parameters['mimic_data_path'] + parameters['multiclassification_directory'] \
                  + parameters['noteevents_hourly_merged_dirname']
if not os.path.exists(new_events_path):
    os.mkdir(new_events_path)

dataset = pd.read_csv(parameters['mimic_data_path'] + parameters['multiclassification_directory']
                      + parameters['all_stays_csv_w_events'])
if 'Unnamed: 0' in dataset.columns:
    dataset = dataset.drop(columns=['Unnamed: 0'])
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
