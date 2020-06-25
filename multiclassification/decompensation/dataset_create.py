import multiprocessing as mp
import os
import sys
from functools import partial

import numpy
import pandas as pd

# TODO: change to comply with icustay_id required in ensemble
def create_episodes(dataset, manager_queue, structured_events_path=None, note_events_path=None,
                     structured_decompensation_events_path=None, textual_decompensation_events_path=None):
    total = len(dataset)
    consumed = 0
    for index, icustay in dataset.iterrows():
        # consumed += 1
        # sys.stderr.write('\rdone {0:%}'.format(consumed / total))
        icustay_id = icustay['ICUSTAY_ID']
        intime = icustay['INTIME']
        outtime = icustay['OUTTIME']
        deathtime = icustay['DEATHTIME']
        icustay_structured_path = os.path.join(structured_decompensation_events_path, str(icustay_id))
        if not os.path.exists(icustay_structured_path):
            os.mkdir(icustay_structured_path)
        icu_structured_events_path = os.path.join(structured_events_path, '{}.csv'.format(icustay_id))
        if not os.path.exists(icu_structured_events_path):
            continue
        structured_events = pd.read_csv(icu_structured_events_path)
        structured_events.loc[:, 'starttime'] = pd.to_datetime(structured_events['starttime'],
                                                               format=parameters['datetime_pattern'])
        textual_events = None
        icustay_textual_path = None
        if os.path.exists(note_events_path + '{}.csv'.format(icustay_id)):
            textual_events = pd.read_csv(os.path.join(note_events_path, '{}.csv'.format(icustay_id)))
            textual_events.loc[:, 'starttime'] = pd.to_datetime(textual_events['starttime'],
                                                                   format=parameters['datetime_pattern'])
            icustay_textual_path = os.path.join(textual_decompensation_events_path, str(icustay_id))
            if not os.path.exists(icustay_textual_path):
                os.mkdir(icustay_textual_path)
        if len(structured_events) >= 4:
            for event_index, event in structured_events.iterrows():
                if event['bucket'] < 3 or event['starttime'] > deathtime:
                    continue
                episode = "{}_{}".format(event['bucket'], icustay_id)
                episode_path = os.path.join(icustay_structured_path, episode + '.csv')
                is_dead_in_24h = 0
                if not pd.isna(deathtime):
                    difference = deathtime - event['starttime']
                    hours = (difference.days * 24) + (difference.seconds // 3600)
                    if hours < 24:
                        is_dead_in_24h = 1
                episode_events = structured_events.loc[:event_index]
                episode_events.to_csv(episode_path)
                manager_queue.put([episode, 'structured', episode_path, is_dead_in_24h])
            if textual_events is not None:
                for event_index, event in textual_events.iterrows():
                    if event['bucket'] < 3 or event['starttime'] > deathtime:
                        continue
                    episode = "{}_{}".format(event['bucket'], icustay_id)
                    is_dead_in_24h = 0
                    if not pd.isna(deathtime):
                        difference = deathtime - event['starttime']
                        hours = (difference.days * 24) + (difference.seconds // 3600)
                        if hours <= 24:
                            is_dead_in_24h = 1
                    episode_events = textual_events.loc[:event_index]
                    episode_path = os.path.join(icustay_textual_path, episode + '.csv')
                    episode_events.to_csv(episode_path)
                    manager_queue.put([episode, 'textual', episode_path, is_dead_in_24h])
    # return manager_queue



from multiclassification.parameters.dataset_parameters import parameters

structured_events_path =  parameters['mimic_data_path'] + parameters['multiclassification_directory'] \
               + parameters['events_hourly_merged_dirname']
note_events_path = parameters['mimic_data_path'] + parameters['multiclassification_directory'] \
                  + parameters['noteevents_hourly_merged_dirname']

structured_decompensation_events_path = parameters['mimic_data_path'] + parameters['multiclassification_directory'] \
                                        + parameters['decompensation_directory'] + parameters['structured_dirname']
if not os.path.exists(structured_decompensation_events_path):
    os.mkdir(structured_decompensation_events_path)
textual_decompensation_events_path = parameters['mimic_data_path'] + parameters['multiclassification_directory'] \
                                        + parameters['decompensation_directory'] + parameters['textual_dirname']
if not os.path.exists(textual_decompensation_events_path):
    os.mkdir(textual_decompensation_events_path)

dataset = pd.read_csv(parameters['mimic_data_path'] + parameters['multiclassification_directory']
                      + parameters['all_stays_csv_w_events'])
if 'Unnamed: 0' in dataset.columns:
    dataset = dataset.drop(columns=['Unnamed: 0'])
dataset.loc[:, 'INTIME'] = pd.to_datetime(dataset['INTIME'], format=parameters['datetime_pattern'])
dataset.loc[:, 'OUTTIME'] = pd.to_datetime(dataset['OUTTIME'], format=parameters['datetime_pattern'])
dataset.loc[:, 'DEATHTIME'] = pd.to_datetime(dataset['DEATHTIME'], format=parameters['datetime_pattern'])
original_len = len(dataset)
total_files = len(dataset)
icustay_ids = list(dataset['ICUSTAY_ID'])
dataset_for_mp = numpy.array_split(dataset, 20)
with mp.Pool(processes=4) as pool:
    m = mp.Manager()
    queue = m.Queue()
    partial_normalize_files = partial(create_episodes, manager_queue=queue,
                                      structured_events_path=structured_events_path,
                                      note_events_path=note_events_path,
                                      structured_decompensation_events_path=structured_decompensation_events_path,
                                      textual_decompensation_events_path=textual_decompensation_events_path)
    # queue = partial_normalize_files(dataset_for_mp)
    map_obj = pool.map_async(partial_normalize_files, dataset_for_mp)
    consumed = 0
    decompensation_dataset = dict()
    while not map_obj.ready():
        for _ in range(queue.qsize()):
            returned_data = queue.get()
            if returned_data[0] not in decompensation_dataset.keys():
                decompensation_dataset[returned_data[0]] = dict()
                decompensation_dataset[returned_data[0]]['episode'] = returned_data[0]
            if returned_data[1] == 'structured':
                decompensation_dataset[returned_data[0]]['structured_path'] = returned_data[2]
                decompensation_dataset[returned_data[0]]['label'] = returned_data[3]
            elif returned_data[1] == 'textual':
                decompensation_dataset[returned_data[0]]['textual_path'] = returned_data[2]
                if 'label' in decompensation_dataset[returned_data[0]].keys():
                    decompensation_dataset[returned_data[0]]['label'] = returned_data[3]
            consumed += 1
            sys.stderr.write('\rConssumed {} events'.format(consumed))
    if queue.qsize() != 0 :
        for _ in range(queue.qsize()):
            returned_data = queue.get()
            if returned_data[0] not in decompensation_dataset.keys():
                decompensation_dataset[returned_data[0]] = dict()
                decompensation_dataset[returned_data[0]]['episode'] = returned_data[0]
            if returned_data[1] == 'structured':
                decompensation_dataset[returned_data[0]]['structured_path'] = returned_data[2]
                decompensation_dataset[returned_data[0]]['label'] = returned_data[3]
            elif returned_data[1] == 'textual':
                decompensation_dataset[returned_data[0]]['textual_path'] = returned_data[2]
                if 'label' in decompensation_dataset[returned_data[0]].keys():
                    decompensation_dataset[returned_data[0]]['label'] = returned_data[3]
    decompensation_dataset = pd.DataFrame(decompensation_dataset)
    decompensation_dataset = decompensation_dataset.T
    decompensation_dataset.to_csv(parameters['mimic_data_path'] + parameters['multiclassification_directory'] \
                                    + parameters['decompensation_directory'] + parameters['decompensation_dataset_csv'])
