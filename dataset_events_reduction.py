"""
This script perform a filtering on the events files that will be used for the classification task
based on the methodology described in (LINK ARTICLE).
For early sepsis detection, is for the best interest that the classifier could perform the detections
with a certain period of anticipation, to give time to provide the adequate treatment.
It will remove events that occur in a window prior to the sofa increasing time. The window varies between 4h-8h, at
steps of 2h. Any patient that don't have at least 3h worth of events after the exclusion of the 8h window.
"""
import csv
from datetime import timedelta

import pandas
import os

import sys


def filter_first_hours(hours, intime, events, column='endtime'):
    delta = timedelta(hours=hours)
    return events[events[column] <= intime + delta]

def filter_last_hours(hours, outtime, events, column='starttime'):
    delta = timedelta(hours=hours)
    filtered_events = events[events[column] >= outtime - delta]
    if len(filtered_events) == 0:
        if len(events) <= hours:
            return events
        else:
            return events[-(hours):]
    return filtered_events

dataset_csv = pandas.read_csv('/home/mattyws/Documents/mimic/new_dataset_patients.csv')
dataset_csv.loc[:, 'intime'] = pandas.to_datetime(dataset_csv['intime'], format="%Y-%m-%d %H:%M:%S")
dataset_csv.loc[:, 'outtime'] = pandas.to_datetime(dataset_csv['outtime'], format="%Y-%m-%d %H:%M:%S")
dataset_csv.loc[:, 'sofa_increasing_time_poe'] = pandas.to_datetime(dataset_csv['sofa_increasing_time_poe'],
                                                                    format="%Y-%m-%d %H:%M:%S")

hours = 24
dataset_path = '/home/mattyws/Documents/mimic/sepsis_articles_bucket/'
textual_dataset_path = '/home/mattyws/Documents/mimic/textual_anonymized_data/'
new_dataset_path = '/home/mattyws/Documents/mimic/sepsis_articles_bucket_first_{}/'.format(hours)
new_textual_dataset_path = '/home/mattyws/Documents/mimic/textual_anonymized_data_first_{}/'.format(hours)
if not os.path.exists(new_dataset_path):
    os.mkdir(new_dataset_path)
if not os.path.exists(new_textual_dataset_path):
    os.mkdir(new_textual_dataset_path)
consumed = 0
total_files = len(dataset_csv)
events_mean = []
for index, row in dataset_csv.iterrows():
    sys.stderr.write('\rdone {0:%}'.format(consumed / total_files))
    if not os.path.exists(new_dataset_path + '{}.csv'.format(row['icustay_id'])):
        events = pandas.read_csv(dataset_path + "{}.csv".format(row['icustay_id']))
        events.loc[:, 'starttime'] = events.loc[:, 'starttime'] = pandas.to_datetime(events['starttime'],
                                                                                         format="%Y-%m-%d %H:%M:%S")
        events.loc[:, 'endtime'] = events.loc[:, 'endtime'] = pandas.to_datetime(events['endtime'],
                                                                                          format="%Y-%m-%d %H:%M:%S")
        hmmm = ""
        if row['class'] == 'sepsis':
            filtered_events = filter_last_hours(hours, row['intime'], events)
            hmmm = 'sepsis'
        else:
            hmmm = 'nonono'
            filtered_events = filter_last_hours(hours, row['intime'], events)
        if len(filtered_events) == 0:
            print(hmmm, row['icustay_id'])
            print(row['sofa_increasing_time_poe'])
            print(row['outtime'])
            print(row['class'])
            print(events[['starttime', 'endtime']])
            exit()
        filtered_events.to_csv(new_dataset_path + '{}.csv'.format(row['icustay_id']), index=False)
        filtered_events_mean = filtered_events.mean()
        filtered_events_mean['icustay'] = row['icustay_id']
        filtered_events_mean['class'] = 'sepsis' if row['class'] == 'sepsis' else 'None'
        events_mean.append(filtered_events_mean)
    else:
        filtered_events = pandas.read_csv(new_dataset_path + '{}.csv'.format(row['icustay_id']))
        filtered_events_mean = filtered_events.mean().to_dict()
        filtered_events_mean['icustay'] = row['icustay_id']
        filtered_events_mean['class'] = 'sepsis' if row['class'] == 'sepsis' else 'None'
        events_mean.append(filtered_events_mean)

    if not os.path.exists(new_textual_dataset_path + '{}.csv'.format(row['icustay_id'])):
        textual_events = pandas.read_csv(textual_dataset_path + "{}.csv".format(row['icustay_id']))
        # print(textual_events)
        textual_events.loc[:, 'charttime'] = textual_events.loc[:, 'charttime'] = pandas.to_datetime(textual_events['charttime'],
                                                                                     format="%Y-%m-%d %H:%M:%S")
        if row['class'] == 'sepsis':
            filtered_events = filter_last_hours(hours, row['intime'], textual_events, column='charttime')
        else:
            filtered_events = filter_last_hours(hours, row['intime'], textual_events, column='charttime')
        filtered_events.to_csv(new_textual_dataset_path + '{}.csv'.format(row['icustay_id']), index=False)

    consumed += 1

# if not os.path.exists('/home/mattyws/Documents/mimic/dataset_mean_last_{}.csv'.format(hours)):
events_mean = pandas.DataFrame(events_mean)
events_mean.to_csv('/home/mattyws/Documents/mimic/dataset_mean_first_{}.csv'.format(hours), na_rep='?',
                   quoting=csv.QUOTE_NONNUMERIC)