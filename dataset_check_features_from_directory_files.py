import os
import pandas as pd
import pprint

import sys

import functions

parameters = functions.load_parameters_file()

directory = parameters['mimic_data_path'] + "sepsis_low_frequency_removed/"
pp = pprint.PrettyPrinter(indent=4)
features_lens = dict()
features = None
listdir = os.listdir(directory)
consumed = 0
for file in listdir:
    events = pd.read_csv(directory + file)
    if features is None:
        features = set(events.columns)
    if len(events.columns) not in features_lens.keys():
        features_lens[len(events.columns)] = 0
    features_lens[len(events.columns)] += 1
    consumed += 1
    sys.stderr.write('\rdone {0:%}'.format(consumed / len(listdir)))

pp.pprint(features_lens)