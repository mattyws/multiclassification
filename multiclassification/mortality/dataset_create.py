import multiprocessing as mp
import os
import sys
from datetime import timedelta
from functools import partial

import numpy
import pandas as pd

from multiclassification import constants


def create_episodes(dataset, manager_queue, hours_since_intime=24, structured_events_path=None, note_events_path=None,
                    structured_mortality_events_path=None, textual_mortality_events_path=None):
    total = len(dataset)
    consumed = 0
    for index, icustay in dataset.iterrows():
        # consumed += 1
        # sys.stderr.write('\rdone {0:%}'.format(consumed / total))
        icustay_id = icustay['ICUSTAY_ID']
        intime = icustay['INTIME']
        outtime = icustay['OUTTIME']
        deathtime = icustay['DEATHTIME']
        label = 0
        los_diff = (outtime - intime)
        if not pd.isna(deathtime):
            label = 1
            los_diff = (deathtime - intime)
        new_structured_events_path = structured_mortality_events_path + '{}.csv'.format(icustay_id)
        new_textual_events_path = textual_mortality_events_path + '{}.csv'.format(icustay_id)
        if os.path.exists(new_structured_events_path):
            manager_queue.put([icustay_id, icustay_id, 'structured', new_structured_events_path, label])
            if os.path.exists(new_textual_events_path):
                manager_queue.put([icustay_id, icustay_id, 'textual', new_textual_events_path, label])
                continue
        los = los_diff.days * 24 + (los_diff.seconds // 3600)
        if los < hours_since_intime:
            continue
        icu_structured_events_path = os.path.join(structured_events_path, '{}.csv'.format(icustay_id))
        if not os.path.exists(icu_structured_events_path):
            continue
        structured_events = pd.read_csv(icu_structured_events_path)
        structured_events.loc[:, 'endtime'] = pd.to_datetime(structured_events['endtime'],
                                                               format=parameters['datetime_pattern'])
        structured_events = structured_events[structured_events['endtime'] <=
                                              intime + timedelta(hours=hours_since_intime)]
        if len(structured_events.dropna(how="all")) == 0:
            continue
        structured_events.to_csv(new_structured_events_path, index=False)
        manager_queue.put([icustay_id, icustay_id, 'structured', new_structured_events_path, label])
        icustay_textual_path = os.path.join(note_events_path, '{}.csv'.format(icustay_id))
        if os.path.exists(icustay_textual_path):
            textual_events = pd.read_csv(icustay_textual_path)
            textual_events.loc[:, 'endtime'] = pd.to_datetime(textual_events['endtime'],
                                                                   format=parameters['datetime_pattern'])
            textual_events = textual_events[textual_events['endtime'] <=
                                                  intime + timedelta(hours=hours_since_intime)]
            empty_rows = []
            bucket_tolist = textual_events['bucket'].tolist()
            for index, srow in structured_events.iterrows():
                if srow['bucket'] not in bucket_tolist:
                    empty_row = dict()
                    empty_row['starttime'] = srow['starttime']
                    empty_row['endtime'] = srow['endtime']
                    empty_row['bucket'] = srow['bucket']
                    empty_row['text'] = constants.NO_TEXT_CONSTANT
                    empty_rows.append(empty_row)
            empty_rows = pd.DataFrame(empty_rows)
            new_textual_events = pd.concat([empty_rows, textual_events], ignore_index=True)
            new_textual_events = new_textual_events.sort_values(by='bucket')
            textual_events = new_textual_events
            textual_events.to_csv(new_textual_events_path, index=False)
            manager_queue.put([icustay_id, icustay_id, 'textual', new_textual_events_path, label])
        else:
            empty_rows = []
            for index, srow in structured_events.iterrows():
                empty_row = dict()
                empty_row['starttime'] = srow['starttime']
                empty_row['endtime'] = srow['endtime']
                empty_row['bucket'] = srow['bucket']
                empty_row['text'] = constants.NO_TEXT_CONSTANT
                empty_rows.append(empty_row)
            empty_rows = pd.DataFrame(empty_rows)
            empty_rows = empty_rows.sort_values(by='bucket')
            empty_rows.to_csv(new_textual_events_path, index=False)
            manager_queue.put([icustay_id, icustay_id, 'textual', new_textual_events_path, label])
    # return manager_queue



from multiclassification.parameters.dataset_parameters import parameters

structured_events_path =  parameters['mimic_data_path'] + parameters['multiclassification_directory'] \
               + parameters['events_hourly_merged_dirname']
note_events_path = parameters['mimic_data_path'] + parameters['multiclassification_directory'] \
                  + parameters['noteevents_hourly_merged_dirname']

structured_decompensation_events_path = parameters['mimic_data_path'] + parameters['multiclassification_directory'] \
                                        + parameters['mortality_directory'] + parameters['structured_dirname']
if not os.path.exists(structured_decompensation_events_path):
    os.mkdir(structured_decompensation_events_path)
textual_decompensation_events_path = parameters['mimic_data_path'] + parameters['multiclassification_directory'] \
                                        + parameters['mortality_directory'] + parameters['textual_dirname']
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
# dataset_for_mp = numpy.array_split(dataset_for_mp[0], 20)
# dataset_for_mp = numpy.array_split(dataset_for_mp[0], 20)
with mp.Pool(processes=4) as pool:
    m = mp.Manager()
    queue = m.Queue()
    partial_normalize_files = partial(create_episodes, manager_queue=queue,
                                      structured_events_path=structured_events_path,
                                      note_events_path=note_events_path,
                                      hours_since_intime=48,
                                      structured_mortality_events_path=structured_decompensation_events_path,
                                      textual_mortality_events_path=textual_decompensation_events_path)
    # queue = partial_normalize_files(dataset_for_mp)
    # partial_normalize_files(dataset_for_mp[0])
    # exit()
    map_obj = pool.map_async(partial_normalize_files, dataset_for_mp)
    consumed = 0
    mortality_dataset = dict()
    while not map_obj.ready():
        for _ in range(queue.qsize()):
            returned_data = queue.get()
            if returned_data[0] not in mortality_dataset.keys():
                mortality_dataset[returned_data[0]] = dict()
                mortality_dataset[returned_data[0]]['episode'] = returned_data[0]
                mortality_dataset[returned_data[0]]['icustay_id'] = returned_data[1]
            if returned_data[2] == 'structured':
                mortality_dataset[returned_data[0]]['structured_path'] = returned_data[3]
                mortality_dataset[returned_data[0]]['label'] = returned_data[4]
            elif returned_data[2] == 'textual':
                mortality_dataset[returned_data[0]]['textual_path'] = returned_data[3]
                if 'label' in mortality_dataset[returned_data[0]].keys():
                    mortality_dataset[returned_data[0]]['label'] = returned_data[4]
            consumed += 1
            sys.stderr.write('\rConssumed {} events'.format(consumed))
    if queue.qsize() != 0 :
        for _ in range(queue.qsize()):
            returned_data = queue.get()
            if returned_data[0] not in mortality_dataset.keys():
                mortality_dataset[returned_data[0]] = dict()
                mortality_dataset[returned_data[0]]['episode'] = returned_data[0]
                mortality_dataset[returned_data[0]]['icustay_id'] = returned_data[1]
            if returned_data[2] == 'structured':
                mortality_dataset[returned_data[0]]['structured_path'] = returned_data[3]
                mortality_dataset[returned_data[0]]['label'] = returned_data[4]
            elif returned_data[2] == 'textual':
                mortality_dataset[returned_data[0]]['textual_path'] = returned_data[3]
                if 'label' in mortality_dataset[returned_data[0]].keys():
                    mortality_dataset[returned_data[0]]['label'] = returned_data[4]
    mortality_dataset = pd.DataFrame(mortality_dataset)
    mortality_dataset = mortality_dataset.T
    mortality_dataset.to_csv(parameters['mimic_data_path'] + parameters['multiclassification_directory'] \
                             + parameters['mortality_directory'] + parameters['mortality_dataset_csv'])
    print(mortality_dataset['label'].value_counts())
    # for index, row in mortality_dataset.iterrows():
    #     events = pd.read_csv(row['structured_path'])
    #     print(len(events))