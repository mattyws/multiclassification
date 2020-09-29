"""
Analyse data using tsfresh package
"""
import pandas as pd
import tsfresh

from resources import functions

parameters = functions.load_parameters_file()
data_path = parameters['mimic_data_path'] + parameters['chartevents_path']
dataset = pd.read_csv(parameters['dataset_file_name'])
print("====== Loading data =======")


