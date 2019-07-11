import multiprocessing as mp
import os
import pickle
import pickle as pkl
import sys
from functools import partial
from itertools import islice

import pandas as pd

# def to_dict_on_saved_file(file):
#     new_data = dict()
#     with open(file, 'rb') as pkl_file:
#         data = pkl.load(pkl_file)
#         for key in data.keys():
#             new_data[key] = data[key].to_dict()
#     with open(file, 'wb') as pkl_file:
#         pkl.dump(new_data, pkl_file)
#
# with mp.Pool(processes=6) as pool:
#     files = ['./normalization_values/' + x for x in os.listdir('./normalization_values/')]
#     for i, result in enumerate(pool.imap(to_dict_on_saved_file, files), 1):
#         sys.stderr.write('\rdone {0:%}'.format(i / len(files)))
from math import ceil


def load_saved_value_count(self, file):
    with open(file, 'rb') as normalization_values_file:
        values = pickle.load(normalization_values_file)
        return values

def cal_sum(lst):
    final_dict = mp.Manager().dict()
    partial_merge_sum_dicts = partial(merge_sum_dicts, final_dict=final_dict)
    for i, l in enumerate(lst, 1):
        counts = load_saved_value_count(l)
        chunks_size = ceil(len(counts.keys())/6)
        counts = [x for x in chunk_dict(counts, SIZE=chunks_size)]
        with mp.Pool(processes=6) as pool:
            pool.map(partial_merge_sum_dicts, counts)
        # self.merge_sum_dicts(final_dict, l)
        sys.stderr.write('\rSum values: done {0:%}'.format(i / len(lst)))
    return final_dict

def chunk_dict(self, data, SIZE=10000):
    it = iter(data)
    for i in range(0, len(data), SIZE):
        yield {k: data[k] for k in islice(it, SIZE)}

def merge_sum_dicts(self, iter_dict, final_dict):
    for k, v in iter_dict.items():
        if isinstance(v, dict):
            sum(final_dict.setdefault(k, dict()), v)
        elif isinstance(v, int):
            final_dict[k] = final_dict.get(k, 0) + v

files = ['./normalization_values/' + x for x in os.listdir('./normalization_values/')][:2]
result = cal_sum(files)
print(result)