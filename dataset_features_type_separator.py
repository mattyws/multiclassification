"""
Separate features with numerical and categorical values
"""
import functions
import pandas as pd

parameters = functions.load_parameters_file()

dataset = pd.read_csv(parameters['mimic_data_path'] + parameters['dataset_file_name'])
patient_events_path = parameters['mimic_data_path'] + "sepsis_raw_merged/"