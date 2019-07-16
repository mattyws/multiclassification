import csv
from functools import partial
from itertools import islice

import numpy as np
import pickle as pkl
import pandas as pd
import os
import multiprocessing as mp

def chunk_lst(data, SIZE=10000):
    it = iter(data)
    for i in range(0, len(data), SIZE):
        yield [k for k in islice(it, SIZE)]

def csv_to_pkl(file_list, file_path, new_path):
    for file in file_list:
        print('### {} ###'.format(file))
        data = pd.read_csv(file_path+file)
        if 'Unnamed: 0' in data.columns:
            data = data.drop(columns=['Unnamed: 0'])
        if 'chartevents_Unnamed: 0' in data.columns:
            data = data.drop(columns=['chartevents_Unnamed: 0'])
        if 'labevents_Unnamed: 0' in data.columns:
            data = data.drop(columns=['labevents_Unnamed: 0'])
        data = np.array(data.values)
        new_file_name = os.path.splitext(file)[0]+'.pkl'
        with open(new_path+new_file_name) as new_file:
            pkl.dump(data, new_file)
        print('### End {} ###'.format(file))

file_path = './data_tmp_0/'
new_path = './new_data_tmp_0/'
if not os.path.exists(new_path):
    os.mkdir(new_path)
data_list = os.listdir(file_path)
data_list = [ x for x in chunk_lst(data_list)]
partial_csv_to_pkl = partial(csv_to_pkl, file_path=file_path, new_path=new_path)
with mp.Pool(processes=6) as pool:
    pool.map(partial_csv_to_pkl, data_list)

new_list = os.listdir(new_path)
with open(new_path+new_list[0], 'rb') as test:
    data = pkl.load(test)
    print(data)