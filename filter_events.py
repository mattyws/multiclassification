import csv
import json
import math
import os
import pprint
import re
import pandas as pd
from datetime import datetime, timedelta

import functions

datetime_pattern = "%Y-%m-%d %H:%M:%S"
features_event_label = 'chartevents'
event_label = 'CHARTEVENTS'
mimic_data_path = "/home/mattyws/Documentos/mimic/data/"
events_csv_path = mimic_data_path + event_label + '/'
events_files_path = mimic_data_path + 'sepsis_{}/'.format(features_event_label)
if not os.path.exists(events_files_path):
    os.mkdir(events_files_path)


json_files_path = mimic_data_path+"json/"
json_files_path = [json_files_path+x+'/' for x in os.listdir(json_files_path)]

sepsis3_df = pd.read_csv('dataset_patients.csv')
# sepsis3_df = sepsis3_df[sepsis3_df["sepsis-3"] == 1]
sepsis3_df['intime'] = pd.to_datetime(sepsis3_df['intime'], format=datetime_pattern)
sepsis3_df['outtime'] = pd.to_datetime(sepsis3_df['outtime'], format=datetime_pattern)
sepsis3_df = sepsis3_df.sort_values(by=['hadm_id', 'intime'])

sepsis3_hadm_ids = sepsis3_df['hadm_id'].values


# Loop through all patients that fits the sepsis 3 definition
for index, row in sepsis3_df.iterrows():
    # If the file isn't found, ignore this admission
    if not os.path.exists(events_csv_path+'{}_{}.csv'.format(event_label, row['hadm_id'])):
        print("File {} do not exists".format(events_csv_path+'{}_{}.csv'.format(event_label, row['hadm_id'])))
        continue

    intime = row['intime']
    outtime = row['outtime']
    events_in_patient = dict()
    # If patient is not healthy either: it fits the sepsis 3 criteria or is getting worse at ICU
    # Either way the events are handled at same manner
    if row['class'] == "sepsis":
        cut_poe = datetime.strptime(row['sofa_increasing_time_poe'], datetime_pattern)
        # Adding marker to infection time
        events_in_patient['-1'] = dict()
        events_in_patient['-1']['itemid'] = -1
        events_in_patient['-1']['label'] = 'sofa_increasing_time_poe'
        events_in_patient['-1'][row['sofa_increasing_time_poe']] = True
    else:
        # If patient is healthy, the cut point will be after 24h of admission at the ICU
        cut_poe = intime + timedelta(hours=24)

    # Loading event csv
    events_df = pd.read_csv(events_csv_path+'{}_{}.csv'.format(event_label, row['hadm_id']))
    # Filter events that occurs between ICU intime and ICU outtime, as the csv corresponds to events that occurs
    # to all hospital admission
    events_df.loc[:, 'CHARTTIME'] = pd.to_datetime(events_df['CHARTTIME'], format=datetime_pattern)
    print("==== Looping events for {} ====".format(row['hadm_id']))
    for index, event in events_df.iterrows():
        if event['CHARTTIME'] >= intime and event['CHARTTIME'] <= cut_poe:
            # Get values and store into a variable, just to read easy and if the labels change
            itemid = event['ITEMID']
            event_timestamp = event['CHARTTIME']
            event_value = event['VALUENUM']
            # If the id is not in events yet, create it and assign a empty dictionary to it
            if itemid not in events_in_patient.keys():
                events_in_patient[itemid] = dict()
                events_in_patient[itemid]['itemid'] = itemid
            # If the timestamp from the event is in the event, assign the higher value between the tow of then
            # It is to check if a same event is masured more than one time at the same timestamp
            if event_timestamp in events_in_patient[itemid].keys():
                if event_value > events_in_patient[itemid][event_timestamp]:
                    events_in_patient[itemid][event_timestamp] = event_value
            else:
                # Else just create the field and assign its value
                events_in_patient[itemid][event_timestamp] = event_value
    print("Converting to dataframe")
    patient_data = pd.DataFrame([])
    for event_id in events_in_patient.keys():
        events = pd.DataFrame(events_in_patient[event_id], index=[0])
        patient_data = pd.concat([patient_data, events], ignore_index=True)
    if len(patient_data) != 0:
        print("==== Creating csv ====")
        patient_data.to_csv(events_files_path + '{}.csv'.format(row['icustay_id']), quoting=csv.QUOTE_NONNUMERIC,
                            index=False)
    else:
        print("Error in file {}, events is empty".format(row['icustay_id']))