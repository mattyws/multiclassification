import os
import pandas as pd
import pprint

import functions

parameters = functions.load_parameters_file()

directory = parameters['mimic_data_path'] + "sepsis_low_frequency_removed/"
pp = pprint.PrettyPrinter(indent=4)
features_lens = dict()
features = None
for file in os.listdir(directory):
    events = pd.read_csv(directory + file)
    if features is None:
        features = set(events.columns)
    if len(events.columns) not in features_lens.keys():
        features_lens[len(events.columns)] = 0
    features_lens[len(events.columns)] += 1

pp.pprint(features_lens)