import csv
import json
from functools import partial
from itertools import product
from multiprocessing.pool import Pool

import math
import os
import pprint
import re
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

import functions

def filter_events(sepsis3_df_spĺit, table_name, mimic_data_path=""):
    events_csv_path = mimic_data_path + table_name + '/'
    filtered_events_file_path = mimic_data_path + 'sepsis_{}/'.format(table_name.lower())
    # Loop through all patients that fits the sepsis 3 definition
    for index, row in sepsis3_df_spĺit.iterrows():
        # Ignore this icustay if already have their events filtered
        filtered_events_file_name = filtered_events_file_path + '{}.csv'.format(row['icustay_id'])
        if os.path.exists(filtered_events_file_name):
            print("Event already filtered for {}".format(row['icustay_id']))
            continue
        # If the file isn't found, ignore this admission
        csv_events_file_name = events_csv_path + '{}_{}.csv'.format(table_name, row['hadm_id'])
        if not os.path.exists(csv_events_file_name):
            print("File {} do not exists".format(events_csv_path + '{}_{}.csv'.format(table_names, row['hadm_id'])))
            continue

        intime = row['intime']
        outtime = row['outtime']
        events_in_patient = dict()
        # If patient is not healthy either: it fits the sepsis 3 criteria or is getting worse at ICU
        # Either way the events are handled at same manner
        if row['class'] == "sepsis" or row['class'] == 'no_infection':
            cut_poe = datetime.strptime(row['sofa_increasing_time_poe'], datetime_pattern)
        else:
            # If patient is healthy, the cut point will be after 24h of admission at the ICU
            cut_poe = intime + timedelta(hours=24)

        # Loading event csv
        events_df = pd.read_csv(csv_events_file_name)
        # Filter events that occurs between ICU intime and ICU outtime, as the csv corresponds to events that occurs
        # to all hospital admission
        events_df.loc[:, 'CHARTTIME'] = pd.to_datetime(events_df['CHARTTIME'], format=datetime_pattern)

        print("==== {} : Looping events for {} ====".format(table_name, row['hadm_id']))
        for index, event in events_df.iterrows():
            # Check if event was an error.
            # As each table has their error columns, we pass the event label to the check function
            # If is a error, pass this event
            if functions.event_is_error(table_name, event):
                continue
            if event['CHARTTIME'] >= intime and event['CHARTTIME'] <= cut_poe:
                event_timestamp = event['CHARTTIME']
                itemid, event_value = functions.get_event_itemid_and_value(table_name, event)
                # If the id is not in events yet, create it and assign a empty dictionary to it
                if itemid not in events_in_patient.keys():
                    events_in_patient[itemid] = dict()
                    events_in_patient[itemid]['itemid'] = itemid
                events_in_patient[itemid][event_timestamp] = event_value
        print("Converting to dataframe")
        patient_data = pd.DataFrame([])
        for event_id in events_in_patient.keys():
            events = pd.DataFrame(events_in_patient[event_id], index=[0])
            patient_data = pd.concat([patient_data, events], ignore_index=True)
        if len(patient_data) != 0:
            print("==== Creating csv ====")
            patient_data = patient_data.set_index(['itemid'])
            patient_data = patient_data.T
            patient_data.to_csv(filtered_events_file_name, quoting=csv.QUOTE_NONNUMERIC)
        else:
            print("Error in file {}, events is empty".format(row['icustay_id']))



parameters = functions.load_parameters_file()

datetime_pattern = parameters['datetime_pattern']
table_names = ['NOTEEVENTS', 'CHARTEVENTS', 'LABEVENTS']
mimic_data_path = parameters['mimic_data_path']
# Creating directories for the filtered events
for table_name in table_names:
    events_files_path = mimic_data_path + 'sepsis_{}/'.format(table_name.lower())
    if not os.path.exists(events_files_path):
        os.mkdir(events_files_path)

sepsis3_df = pd.read_csv(parameters["dataset_file_name"])
# sepsis3_df = sepsis3_df[sepsis3_df["sepsis-3"] == 1]
sepsis3_df['intime'] = pd.to_datetime(sepsis3_df['intime'], format=datetime_pattern)
sepsis3_df['outtime'] = pd.to_datetime(sepsis3_df['outtime'], format=datetime_pattern)
sepsis3_df = sepsis3_df.sort_values(by=['hadm_id', 'intime'])

sepsis3_hadm_ids = sepsis3_df['hadm_id'].values

partial_filter_events = partial(filter_events,
                          mimic_data_path=mimic_data_path)
with Pool(processes=4) as pool:
    df_split = np.array_split(sepsis3_df, 10)
    product_parameters = product(df_split, table_names)
    pool.starmap(partial_filter_events, product_parameters)