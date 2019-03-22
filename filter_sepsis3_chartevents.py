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
mimic_data_path = "/home/mattyws/Documentos/mimic/data/"
events_files_path = mimic_data_path + 'sepsis_{}/'.format(features_event_label)
no_sepsis_events_files_path = mimic_data_path + 'no_sepsis_{}/'.format(features_event_label)
if not os.path.exists(no_sepsis_events_files_path):
    os.mkdir(no_sepsis_events_files_path)

if not os.path.exists(events_files_path):
    os.mkdir(events_files_path)


json_files_path = mimic_data_path+"json/"
json_files_path = [json_files_path+x+'/' for x in os.listdir(json_files_path)]

sepsis3_df = pd.read_csv('sepsis3-df-no-exclusions.csv')
sepsis3_df = sepsis3_df[sepsis3_df["sepsis-3"] == 1]
sepsis3_df['intime'] = pd.to_datetime(sepsis3_df['intime'], format=datetime_pattern)
sepsis3_df = sepsis3_df.sort_values(by=['hadm_id', 'intime'])

sepsis3_hadm_ids = sepsis3_df['hadm_id'].values

print(" /** Get sepsis files paths **/")
sepsis_admissions_paths = functions.get_files_by_ids(sepsis3_hadm_ids, json_files_path)

# Loop through all patients that fits the sepsis 3 definition
for index, row in sepsis3_df.iterrows():
    # If the file isn't found, ignore this admission
    if sepsis_admissions_paths[row['hadm_id']]:
        # Open the json file, transform the date to a datetime representation, and calculate the difference
        json_admission = json.load(open(sepsis_admissions_paths[row['hadm_id']], 'r'))
        # If the admission was already processed, there is no reason to add it again
        if os.path.exists(events_files_path +'{}.csv'.format(row['hadm_id'])):
            print("{} already exists!".format(events_files_path +'{}.csv'.format(row['hadm_id'])))
            continue
        if features_event_label not in json_admission:
            continue
        admittime_datetime = datetime.strptime(json_admission['admittime'], datetime_pattern)
        infection_datetime = datetime.strptime(row['suspected_infection_time_poe'], datetime_pattern)
        diff = infection_datetime - admittime_datetime
        # If passed 1+ day since the admission, it will be generated a representation of the data based on the events_label
        if diff.days > 0:
            # Create a dictionary of events that occur in the patient, and add all timestamps where that event appear
            events_in_patient = dict()
            # Adding marker to infection time
            events_in_patient['-1'] = dict()
            events_in_patient['-1']['itemid'] = -1
            events_in_patient['-1']['label'] = 'suspected_infection_time_poe'
            events_in_patient['-1'][row['suspected_infection_time_poe']] = True
            print("==== Looping events for {} ====".format(row['hadm_id']))
            for item in json_admission[features_event_label]:
                # Get values and store into a variable, just to read easy and if the labels change
                itemid = item['ITEMID']
                item_label = item['ITEM']
                event_timestamp = item['charttime']
                event_value = item['value']
                # Check if event occurs after infection_datetime, and if it is, ignore this event
                event_datetime = datetime.strptime(event_timestamp, datetime_pattern)
                if event_datetime > infection_datetime:
                    continue
                # If the id is not in events yet, create it and assign a empty dictionary to it
                if itemid not in events_in_patient.keys():
                    events_in_patient[itemid] = dict()
                    events_in_patient[itemid]['itemid'] = itemid
                    events_in_patient[itemid]['label'] = item_label
                # If the timestamp from the event is in the event, assign the higher value between the tow of then
                # It is to check if a same event is masured more than one time at the same timestamp
                if event_timestamp in events_in_patient[itemid].keys():
                    if event_value > events_in_patient[itemid][event_timestamp]:
                        events_in_patient[itemid][event_timestamp] = event_value
                else:
                    # Else just create the field and assign its value
                    events_in_patient[itemid][event_timestamp] = event_value
            # Now creating a dataframe to make it easier to handle
            print("Converting to dataframe")
            patient_data = pd.DataFrame([])
            for event_id in events_in_patient.keys():
                events = pd.DataFrame(events_in_patient[event_id], index=[0])
                patient_data = pd.concat([patient_data, events], ignore_index=True)
            print("==== Creating csv ====")
            patient_data.to_csv(events_files_path +'{}.csv'.format(row['hadm_id']), quoting=csv.QUOTE_NONNUMERIC, index=False)
    else:
        print("Admission {} not found: ".format(row['hadm_id']))


sepsis3_df = pd.read_csv('sepsis3-df-no-exclusions.csv')
no_sepsis3_df = sepsis3_df[sepsis3_df['suspicion_poe'] == True]
no_sepsis3_df = no_sepsis3_df[no_sepsis3_df['sepsis-3'] == 0]

print("/** Get admissions without sepsis **/")
no_sepsis3_hadm_ids = no_sepsis3_df['hadm_id'].values
no_sepsis_admissions_paths = functions.get_files_by_ids(no_sepsis3_hadm_ids, json_files_path)

# Get admissions that do not has sepsis marked
for index, row in no_sepsis3_df.iterrows():
    # If the file isn't found, ignore this admission
    if no_sepsis_admissions_paths[row['hadm_id']]:
        try:
            json_admission = json.load(open(no_sepsis_admissions_paths[row['hadm_id']], 'r'))
        except:
            print("Error in file {}".format(no_sepsis_admissions_paths[row['hadm_id']]))
            exit(1)
        if os.path.exists(no_sepsis_events_files_path + '{}.csv'.format(json_admission['hadm_id'])):
            print("{} already exists!".format(no_sepsis_events_files_path + '{}.csv'.format(json_admission['hadm_id'])))
            continue
        if json_admission['hadm_id'] not in sepsis3_hadm_ids and features_event_label in json_admission.keys():
            admittime_datetime = datetime.strptime(json_admission['admittime'], datetime_pattern)
            admittime_datetime_plus1day = admittime_datetime + timedelta(days=1)
            # Create a dictionary of events that occur in the patient, and add all timestamps where that event appear
            events_in_patient = dict()
            print("==== Looping events for {} ====".format(json_admission['hadm_id']))
            for item in json_admission[features_event_label]:
                # Get values and store into a variable, just to read easy and if the labels change
                itemid = item['ITEMID']
                item_label = item['ITEM']
                event_timestamp = item['charttime']
                event_value = item['value']
                # Check if event occurs after one day of admission, if its true, ignore the event
                event_datetime = datetime.strptime(event_timestamp, datetime_pattern)
                if event_datetime > admittime_datetime_plus1day:
                    continue
                # If the id is not in events yet, create it and assign a empty dictionary to it
                if itemid not in events_in_patient.keys():
                    events_in_patient[itemid] = dict()
                    events_in_patient[itemid]['itemid'] = itemid
                    events_in_patient[itemid]['label'] = item_label
                # If the timestamp from the event is in the event, assign the higher value between the tow of then
                # It is to check if a same event is masured more than one time at the same timestamp
                if event_timestamp in events_in_patient[itemid].keys():
                    if event_value > events_in_patient[itemid][event_timestamp]:
                        events_in_patient[itemid][event_timestamp] = event_value
                else:
                    # Else just create the field and assign its value
                    events_in_patient[itemid][event_timestamp] = event_value
            # Now creating a dataframe to make it easier to handle
            print("Converting to dataframe")
            patient_data = pd.DataFrame([])
            for event_id in events_in_patient.keys():
                events = pd.DataFrame(events_in_patient[event_id], index=[0])
                patient_data = pd.concat([patient_data, events], ignore_index=True)
            print("==== Creating csv ====")
            patient_data.to_csv(no_sepsis_events_files_path + '{}.csv'.format(json_admission['hadm_id']), quoting=csv.QUOTE_NONNUMERIC,
                                index=False)
