import csv
import os
import pickle
from functools import partial
from itertools import islice

import math

import pandas as pd
import numpy as np
import multiprocessing as mp

import sys

from math import ceil


def get_file_value_counts(file, pickle_object_path):
    """
    Get the values count for a csv file, and save the result into a pickle file, based the path parameter
    :param file: the file to get the counts
    :param pickle_object_path: the path to store the value counts
    :return:
    """
    pickle_fname = file.split('/')[-1]
    pickle_fname = pickle_object_path + pickle_fname.split('.')[0] + '.pkl'
    if os.path.exists(pickle_fname):
        # File already exists, do not create it
        return file, pickle_fname
    if file == None or (file != None and len(file) == 0):
        return None
    df = pd.read_csv(file)
    if 'Unnamed: 0' in df.columns:
        df = df.set_index(['Unnamed: 0'])
    if 'labevents_Unnamed: 0' in df.columns:
        df = df.drop(columns=['labevents_Unnamed: 0'])
    counts = dict()
    for column in df.columns:
        counts[column] = df[column].value_counts().to_dict()
        # counts[column].index = counts[column].index.map(float)
    try:
        with open(pickle_fname, 'wb') as result_file:
            pickle.dump(counts, result_file)
        return file, pickle_fname
    except Exception as e:
        print("Some error happen on {}. Exception {}".format(file, e))


def get_saved_value_count(file):
    with open(file, 'rb') as normalization_values_file:
        values = pickle.load(normalization_values_file)
        return values

def chunk_lst(data, SIZE=10000):
    it = iter(data)
    for i in range(0, len(data), SIZE):
        yield [k for k in islice(it, SIZE)]


class NormalizationValues(object):
    def __init__(self, files_list, pickle_object_path="normalization_values/"):
        self.files_list = files_list
        if pickle_object_path[-1] != '/':
            pickle_object_path += '/'
        if not os.path.exists(pickle_object_path):
            os.mkdir(pickle_object_path)
        self.get_file_value_counts = partial(get_file_value_counts, pickle_object_path=pickle_object_path)
        self.counts = None

    def prepare(self):
        """
        Prepare the data getting their value counts for each column, and saving the count object into a pickle file
        :return:
        """
        self.counts = dict()
        with mp.Pool(processes=6) as pool:
            for i, result in enumerate(pool.imap(self.get_file_value_counts, self.files_list), 1):
                sys.stderr.write('\rdone {0:%}'.format(i / len(self.files_list)))
                if result is not None:
                    self.counts[result[0]] = result[1]
            print()

    def get_normalization_values(self, training_files):
        """
        Get the max, min, mean and std value for each column from a set of csv files used for training the model
        :return: a dict with the values for each column
        """
        fnames = []
        for file in training_files:
            fnames.append(self.counts[file])
        values = self.sum_counts(fnames)
        new_values = dict()
        for key in values.keys():
            new_values[key] = dict()
            unique_values = list(values[key].keys())
            count_values = [values[key][k] for k in unique_values]
            new_values[key]['max'] = max(unique_values)
            new_values[key]['min'] = min(unique_values)
            mean_std = self.__weighted_avg_and_std(unique_values, count_values)
            new_values[key]['mean'] = mean_std[0]
            new_values[key]['std'] = mean_std[1]
        return new_values

    def sum_counts(self, lst):
        total_files = len(lst)
        chunks_size = ceil(len(lst) / 10)
        lst = [x for x in chunk_lst(lst, SIZE=chunks_size)]
        with mp.Pool(processes=len(lst)) as pool:
            m = mp.Manager()
            queue = m.Queue()
            partial_seq_cal_sum = partial(self.cal_sum, queue=queue)
            map_obj = pool.map_async(partial_seq_cal_sum, lst)
            consumed = 0
            while not map_obj.ready():
                for _ in range(queue.qsize()):
                    queue.get()
                    consumed += 1
                sys.stderr.write('\rdone {0:%}'.format(consumed / total_files))
            result = map_obj.get()
            print()
        result = self.cal_sum(result, list_dicts=True)
        return result

    def cal_sum(self, lst, list_dicts=False, queue=None):
        final_dict = dict()
        for l in lst:
            if list_dicts:
                counts = l
            else:
                counts = self.__load_saved_value_count(l)
            self.merge_sum_dicts(counts, final_dict)
            if queue is not None:
                queue.put(l)
        return final_dict

    def merge_sum_dicts(self, iter_dict, final_dict):
        new_dict = {}
        new_dict.update(final_dict)
        for key in iter_dict.keys():
            new_dict.setdefault(key, dict())
            for k, v in iter_dict[key].items():
                new_dict[key][k] = new_dict[key].get(k, 0) + v
        final_dict.update(new_dict)

    def __load_saved_value_count(self, file):
        with open(file, 'rb') as normalization_values_file:
            values = pickle.load(normalization_values_file)
            return values

    def __weighted_avg_and_std(self, values, weights):
        """
        Return the weighted average and standard deviation.

        values, weights -- Numpy ndarrays with the same shape.
        """
        average = np.average(values, weights=weights)
        # Fast and numerically precise:
        variance = np.average((values - average) ** 2, weights=weights)
        return (average, math.sqrt(variance))


class Normalization(object):

    def __init__(self, normalization_values, temporary_path='./data_tmp/'):
        self.normalization_values = normalization_values
        self.temporary_path = temporary_path
        if not os.path.exists(temporary_path):
            os.mkdir(temporary_path)
        if not temporary_path.endswith('/'):
            temporary_path += '/'
        self.new_paths = None

    def normalize_files(self, filesList):
        """
        Normalize all files in a list of paths
        :param filesList: the list of files path
        :return: a new list for the paths of the normalized data
        """
        total_files = len(filesList)
        chunks_size = ceil(len(filesList)/10)
        filesList = [x for x in chunk_lst(filesList, SIZE=chunks_size)]
        self.new_paths = dict()
        with mp.Pool(processes=len(filesList)) as pool:
            m = mp.Manager()
            queue = m.Queue()
            partial_normalize_files = partial(self.__normalize_files, queue=queue)
            map_obj = pool.map_async(partial_normalize_files, filesList)
            consumed = 0
            while not map_obj.ready():
                for _ in range(queue.qsize()):
                    queue.get()
                    consumed += 1
                sys.stderr.write('\rdone {0:%}'.format(consumed / total_files))
            result = map_obj.get()
            print()
            for r in result:
                self.new_paths.update(r)

    def __normalize_files(self, files_list, queue=None):
        new_paths = dict()
        for l in files_list:
            pair = self.__normalize_file(l)
            new_paths[pair[0]] = pair[1]
            if queue is not None:
                queue.put(l)
        return new_paths


    def get_new_paths(self, files_list):
        if self.new_paths is not None:
            new_list = []
            for file in files_list:
                new_list.append(self.new_paths[file])
            return new_list
        else:
            raise Exception("Data not normalized!")

    def __normalize_file(self, file):
        fileName = file.split('/')[-1]
        data = pd.read_csv(file)
        if 'Unnamed: 0' in data.columns:
            data = data.drop(columns=['Unnamed: 0'])
        if 'labevents_Unnamed: 0' in data.columns:
            data = data.drop(columns=['labevents_Unnamed: 0'])
        data = self.__normalize_dataframe(data, self.normalization_values)
        data.to_csv(self.temporary_path + fileName, index=False)
        return file, self.temporary_path + fileName

    def __normalize_dataframe(self, data, normalization_values):
        """
        Normalize data using the normalization_values (max and min for the column)
        :param data: the data to be normalized
        :return: the data normalized
        """
        for column in data.columns:
            data.loc[:, column] = self.__z_score_normalization(column, data[column], normalization_values)
        return data

    def __min_max_normalization(self, column, series, normalization_values):
        max = normalization_values[column]['max']
        min = normalization_values[column]['min']
        return series.apply(lambda x: (x - min) / (max - min))

    def __z_score_normalization(self, column, series, normalization_values):
        # If std is equal to 0, all columns have the same value
        if normalization_values[column]['std'] != 0:
            mean = normalization_values[column]['mean']
            std = normalization_values[column]['std']
            return series.apply(lambda x: (x - mean) / std)
        return series
