import multiprocessing as mp
import os
import pickle
import pickle as pkl
import sys
from functools import partial
from itertools import islice
from multiprocessing import Lock, Process
from time import time

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

class Counter(object):
    def __init__(self):
        self.val = mp.Value('i', 0)

    def increment(self, n=1):
        with self.val.get_lock():
            self.val.value += n

    @property
    def value(self):
        return self.val.value

def load_saved_value_count(file):
    with open(file, 'rb') as normalization_values_file:
        values = pickle.load(normalization_values_file)
        return values

def sum_counts(lst):
    total_files = len(lst)
    chunks_size = ceil(len(lst)/10)
    lst = [x for x in chunk_lst(lst, SIZE=chunks_size)]
    # processes = []
    # for chunk in lst:
    #     p = Process(target=seq_cal_sum, args=(chunk))
    #     p.start()
    #     processes.append(p)
    # for p in processes:
    #     p.join()
    with mp.Pool(processes=len(lst)) as pool:
        m = mp.Manager()
        queue = m.Queue()
        partial_seq_cal_sum = partial(seq_cal_sum, queue=queue)
        map_obj = pool.map_async(partial_seq_cal_sum, lst)
        consumed = 0
        while not map_obj.ready():
            for _ in range(queue.qsize()):
                queue.get()
                consumed += 1
            sys.stderr.write('\rdone {0:%}'.format(consumed / total_files))
        result = map_obj.get()
    result = seq_cal_sum(result, list_dicts=True)
    return result


def chunk_lst(data, SIZE=10000):
    it = iter(data)
    for i in range(0, len(data), SIZE):
        yield [k for k in islice(it, SIZE)]

def seq_cal_sum(lst, list_dicts=False, queue=None):
    final_dict = dict()
    for l in lst:
        if list_dicts:
            counts = l
        else:
            counts = load_saved_value_count(l)
        merge_sum_dicts(counts, final_dict)
        if queue is not None:
            queue.put(l)
    return final_dict

def seq_merge_sum_dicts(iter_dict, final_dict):
    for k, v in iter_dict.items():
        if isinstance(v, dict):
            seq_merge_sum_dicts(final_dict.setdefault(k, dict()), v)
        elif isinstance(v, int):
            final_dict[k] = final_dict.get(k, 0) + v

def merge_sum_dicts(iter_dict, final_dict):
    new_dict = {}
    new_dict.update(final_dict)
    for key in iter_dict.keys():
        new_dict.setdefault(key, dict())
        for k, v in iter_dict[key].items():
            new_dict[key][k] = new_dict[key].get(k, 0) + v
    # for k, v in iter_dict.items():
    #     if isinstance(v, dict):
    #         # new_dict[k] = new_dict.get(k, dict())
    #         merge_sum_dicts(v, new_dict.setdefault(k, dict()))
    #     elif isinstance(v, float) or isinstance(v, int):
    #         new_dict[k] = new_dict.get(k, 0) + v
    final_dict.update(new_dict)

files = ['./normalization_values/' + x for x in os.listdir('./normalization_values/')][:6000]
print(len(files))
print("Multiprocessing ############################")
start = time()
result = sum_counts(files)
end = time()
print(result.keys())
print(len(result.keys()))
print("{} seconds for multiprocess".format(end-start))
# print("Sequential ############################")
# start = time()
# result = seq_cal_sum(files)
# end = time()
# print(result)
# print("{} seconds for sequential".format(end-start))
# print()
# print(len(result.keys()))
# print(result)