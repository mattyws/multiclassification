import json
import logging
import os
import pandas as pd
from adapter import Word2VecTrainer
from data_generators import LengthLongitudinalDataGenerator, NoteeventsTextDataGenerator
from functions import load_parameters_file

parameters = load_parameters_file()

# Loading csv
print("========= Loading data")
data_csv = pd.read_csv(parameters['mimic_data_path'] + parameters['dataset_file_name'])
data_csv = data_csv.sort_values(['icustay_id'])
positives = data_csv[data_csv['class'] == 'sepsis']
negatives = data_csv[data_csv['class'] != 'sepsis']

len_pos = (len(positives) * 1.6)
negatives = negatives.sample(n=int(len_pos))
data_csv = positives.append(negatives)
data_csv.to_csv(parameters['mimic_data_path'] + parameters['dataset_file_undersampled_name'])