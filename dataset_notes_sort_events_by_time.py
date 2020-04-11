import pandas as pd
import os

import sys

textual_data_path = "../mimic/textual_anonymized_data/"
patients = [textual_data_path + x for x in os.listdir(textual_data_path)]
datetime_pattern = "%Y-%m-%d %H:%M:%S"
total_files = len(patients)
consumed = 0
for patient in patients:
    sys.stderr.write('\rdone {0:%}'.format(consumed / total_files))
    events = pd.read_csv(patient)
    events['charttime'] = pd.to_datetime(events['Unnamed: 0'], format=datetime_pattern)
    events = events.drop(columns=['Unnamed: 0'])
    events = events.sort_values(by=['charttime'])
    events.to_csv(patient)
    consumed += 1
