import multiprocessing as mp
import os
import pickle as pkl
import sys
from functools import partial

import pandas as pd

def to_dict_on_saved_file(file):
    new_data = dict()
    with open(file, 'rb') as pkl_file:
        data = pkl.load(pkl_file)
        for key in data.keys():
            new_data[key] = data[key].to_dict()
    with open(file, 'wb') as pkl_file:
        pkl.dump(new_data, pkl_file)

with mp.Pool(processes=6) as pool:
    files = ['./normalization_values/' + x for x in os.listdir('./normalization_values/')]
    for i, result in enumerate(pool.imap(to_dict_on_saved_file, files), 1):
        sys.stderr.write('\rdone {0:%}'.format(i / len(files)))