"""
Filter selected features ids in mimic dataset and gets statistical values about the filtered features
"""
import pandas as pd

import functions
import os

# def filter_features(files_list, features)

parameters = functions.load_parameters_file()

dataset_merged_files_path = parameters['mimic_data_path'] + 'sepsis_chartevents/'
dataset_filtered_files_path = parameters['mimic_data_path'] + parameters['raw_events_dirname'].format('filtered')

events_ids = {
    "systolic_blood_pressure" : [6, 51, 442, 455, 3315, 3317, 3321, 3323, 6701, 224167, 227243, 220050, 220179, 225309],
    "diastolic_blood_pressure" : [5364, 8368, 8440, 8441, 8503, 8504, 8506, 8507, 8555, 227242, 224643, 220051, 220180,
                                  225310],
    "temperature_fahrenheit" : [676, 677, 3655, 226329, 223762],
    "temperature_celcius" : [678, 679, 3652, 3654, 223761],
    "respiratory_rate" : [614, 615, 618, 619, 653, 1884, 8113, 3603, 224688, 224689, 224690, 220210],
    "glucose": [1445, 1310, 807, 811, 3744, 3745, 1529, 2338, 2416, 228388, 225664, 220621, 226537, 5431],
    "heart_rate": [211, 220045],
    "blood_oxygen_saturation": [50817]
}

files_list = [dataset_merged_files_path + x for x in os.listdir(dataset_merged_files_path)]

all_ids = []
for key in events_ids.keys():
    all_ids.extend(events_ids[key])
for f in files_list:
    patient_events = pd.read_csv(f)
    ids_in_patients = ['Unnamed: 0']
    for column in patient_events.columns:
        if column != 'Unnamed: 0' and int(column) in all_ids:
            ids_in_patients.append(column)
    patient_events = patient_events[ids_in_patients]
