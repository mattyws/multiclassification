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
mimic_data_path = "/home/mattyws/Documents/"
events_files_path = mimic_data_path + 'sepsis_{}/'.format(features_event_label)

if not os.path.exists(events_files_path):
    os.mkdir(events_files_path)


# json_files_path = mimic_data_path+"json/"
# json_files_path = [json_files_path+x+'/' for x in os.listdir(json_files_path)]

sepsis3_df = pd.read_csv('sepsis3-df-no-exclusions.csv')
sepsis3_df = sepsis3_df[sepsis3_df["sepsis-3"] == 1]
sepsis3_df['intime'] = pd.to_datetime(sepsis3_df['intime'], format=datetime_pattern)
sepsis3_df = sepsis3_df.sort_values(by=['hadm_id', 'intime'])

hadm_ids = sepsis3_df['hadm_id'].values

# print(" /** Get sepsis files paths **/")
# sepsis_admissions_paths = functions.get_files_by_ids(hadm_ids, json_files_path)
#
# # Loop through all patients that fits the sepsis 3 definition
# for index, row in sepsis3_df.iterrows():
#     # If the file isn't found, ignore this admission
#     if sepsis_admissions_paths[row['hadm_id']]:
#         # Open the json file, transform the date to a datetime representation, and calculate the difference
#         json_admission = json.load(open(sepsis_admissions_paths[row['hadm_id']], 'r'))
#         if os.path.exists(events_files_path +'{}.csv'.format(row['hadm_id'])):
#             continue
#         if features_event_label not in json_admission:
#             continue
#         admittime_datetime = datetime.strptime(json_admission['admittime'], datetime_pattern)
#         infection_datetime = datetime.strptime(row['suspected_infection_time_poe'], datetime_pattern)
#         diff = infection_datetime - admittime_datetime
#         # If passed 1+ day since the admission, it will be generated a representation of the data based on the events_label
#         if diff.days > 0:
#             # Create a dictionary of events that occur in the patient, and add all timestamps where that event appear
#             events_in_patient = dict()
#             print("==== Looping events for {} ====".format(row['hadm_id']))
#             for item in json_admission[features_event_label]:
#                 # Get values and store into a variable, just to read easy and if the labels change
#                 itemid = item['ITEMID']
#                 item_label = item['ITEM']
#                 event_timestamp = item['charttime']
#                 event_value = item['value']
#                 # If the id is not in events yet, create it and assign a empty dictionary to it
#                 if itemid not in events_in_patient.keys():
#                     events_in_patient[itemid] = dict()
#                     events_in_patient[itemid]['itemid'] = itemid
#                     events_in_patient[itemid]['label'] = item_label
#                 # If the timestamp from the event is in the event, assign the higher value between the tow of then
#                 # It is to check if a same event is masured more than one time at the same timestamp
#                 if event_timestamp in events_in_patient[itemid].keys():
#                     if event_value > events_in_patient[itemid][event_timestamp]:
#                         events_in_patient[itemid][event_timestamp] = event_value
#                 else:
#                     # Else just create the field and assign its value
#                     events_in_patient[itemid][event_timestamp] = event_value
#             # Now creating a dataframe to make it easier to handle
#             print("Converting to dataframe")
#             patient_data = pd.DataFrame([])
#             for event_id in events_in_patient.keys():
#                 events = pd.DataFrame(events_in_patient[event_id], index=[0])
#                 patient_data = pd.concat([patient_data, events], ignore_index=True)
#             print("==== Creating csv ====")
#             patient_data.to_csv(events_files_path +'{}.csv'.format(row['hadm_id']), quoting=csv.QUOTE_NONNUMERIC, index=False)
#     else:
#         print("Admission {} not found: ".format(row['hadm_id']))


# TODO : remover eventos depois do tempo de suspeita de infecção
# TODO : rodar novamente o filtro pois não é possível fazer devido a não ter a hora de adimissão
csv_files_paths = [events_files_path+x for x in os.listdir(events_files_path)]
ids_in_directory = [x.split('/')[-1].split('.')[0] for x in os.listdir(events_files_path)]
for path, id in zip(csv_files_paths, ids_in_directory):
    print(path, id)
    sepsis3_id = sepsis3_df[sepsis3_df['hadm_id'] == int(id)]
    sepsis3_id = sepsis3_id.sort_values(by=['intime'])
    print(sepsis3_id['intime'])
    data = pd.read_csv(path)
    columns = data.columns
    for col in columns:
        try:
            col_datetime = datetime.strptime(col, datetime_pattern)

        except:
            print(col)
            continue