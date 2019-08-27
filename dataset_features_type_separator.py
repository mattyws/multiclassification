"""
Separate features with numerical and categorical values
"""
import os
import pickle

import functions
import pandas as pd

parameters = functions.load_parameters_file()

dataset = pd.read_csv(parameters['mimic_data_path'] + parameters['dataset_file_name'])
patient_events_path = parameters['mimic_data_path'] + "sepsis_raw_merged/"

if not os.path.exists(parameters['mimic_data_path'] + parameters['features_types_file_name']):
    for icustay in dataset['icustay_id']:
        events = pd.read_csv(patient_events_path + "{}.csv".format(icustay))

else:
    features_types = pickle.load(parameters['mimic_data_path'] + parameters['features_types_file_name'])