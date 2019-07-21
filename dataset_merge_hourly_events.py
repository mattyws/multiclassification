"""
This script will merge all events that occur at the same hour (taking patient's intime at the icu as the start),
into one events, using the mean of the values at this window.
"""
from itertools import islice
from math import ceil

import pandas as pd
import multiprocessing as mp

import functions


def process_events(dataset, events_path, hours_prior_sofa=[4, 6, 8], datetime_pattem='%Y-%m-%d %H:%M:%S',
                   manager_queue=None):
    for index, patient in dataset.iterrows():
        events = pd.read_csv(events_path+'{}.csv'.format(patient['icustay_id']))
        # Unnamed: 0 here represents the time for the events




def chunk_lst(data, SIZE=10000):
    it = iter(data)
    for i in range(0, len(data), SIZE):
        yield [k for k in islice(it, SIZE)]

parameters = functions.load_parameters_file()

dataset = pd.read_csv(parameters['dataset_file_name'])
if 'Unnamed: 0' in dataset.columns:
    dataset = dataset.drop('Unnamed: 0')
dataset.iloc[:, 'intime'] = pd.to_datetime(dataset['intime'], format=parameters['datetime_pattern'])
dataset.iloc[:, 'sofa_increasing_time_poe'] = pd.to_datetime(dataset['intime'],
                                                             format=parameters['sofa_increasing_time_poe'])
icustay_ids = dataset['icustay_id'].to_list()
icustay_ids = [x for x in chunk_lst(icustay_ids, SIZE=ceil(len(icustay_ids)/6))]
dataset_for_mp = [ dataset[dataset['icustay_id'].isin(x)] for x in icustay_ids ]

