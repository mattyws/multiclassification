import multiprocessing as mp
import os
import pickle as pkl
import sys
from functools import partial

import pandas as pd

# TODO : paralelization
def cal_sum(lst):
    manager = mp.Manager()
    final_dict = manager.dict()
    # final_dict = dict()
    for l in lst:
        sum(final_dict,l)
    print(final_dict)
    return final_dict

def sum(final_dict,iter_dict):
    pairs = [(k, v) for k, v in iter_dict.items()]
    partial_real_sum = partial(real_sum, final_dict=final_dict)
    with mp.Pool(processes=6) as pool:
        pool.map(partial_real_sum, pairs)
    # for k, v in iter_dict.items():
    #     if isinstance(v, dict):
    #         sum(final_dict.setdefault(k, dict()), v)
    #     elif isinstance(v, int):
    #         final_dict[k] = final_dict.get(k, 0) + v

def real_sum(pair, final_dict):
    k, v = pair
    print(type(pair), type(k), type(v))
    if isinstance(v, dict):
        sum(final_dict.setdefault(k, dict()), v)
    elif isinstance(v, int):
        final_dict[k] = final_dict.get(k, 0) + v

manager = mp.Manager()
manager_dict = manager.dict()



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