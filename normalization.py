import csv
import os
import pickle
from functools import partial

import math

import pandas as pd
import numpy as np
import multiprocessing as mp

import sys


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

## Sum a list of dicts
# TODO : paralelization
def cal_sum(lst):
    final_dict = dict()
    for l in lst:
        sum(final_dict,l)
    return final_dict

def sum(final_dict,iter_dict):
    for k, v in iter_dict.items():
        if isinstance(v, dict):
            sum(final_dict.setdefault(k, dict()), v)
        elif isinstance(v, int):
            final_dict[k] = final_dict.get(k, 0) + v


class NormalizationValues(object):
    def __init__(self, files_list, pickle_object_path="normalization_values/"):
        self.files_list = files_list
        if pickle_object_path[-1] != '/':
            pickle_object_path += '/'
        if not os.path.exists(pickle_object_path):
            os.mkdir(pickle_object_path)
        self.get_file_value_counts = partial(get_file_value_counts, pickle_object_path=pickle_object_path)
        self.get_saved_value_count = get_saved_value_count
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
        values = []
        # Loop each file in dataset
        with mp.Pool(processes=6) as pool:
            for i, result in enumerate(pool.imap(self.get_saved_value_count, training_files), 1):
                sys.stderr.write('\rLoading files: Done {0:%}'.format(i / len(self.files_list)))
                if result is not None:
                    values.append(result)
            print()
        for i, file in enumerate(training_files, 1):
            sys.stderr.write('\rLoading files: done {0:%}'.format(i / len(training_files)))
            file_value_count = self.__get_saved_value_count(self.counts[file])
            if values is None:
                values = file_value_count
            else:
                for key in values.keys():
                    values[key] = values[key].combine(file_value_count[key], func = (lambda x1, x2: x1 + x2),
                                                      fill_value=0.0)
        new_values = dict()
        for key in values.keys():
            new_values[key] = dict()
            new_values[key]['max'] = values[key].index.max()
            new_values[key]['min'] = values[key].index.min()
            mean_std = self.__weighted_avg_and_std(values[key].index, values[key].values)
            new_values[key]['mean'] = mean_std[0]
            new_values[key]['std'] = mean_std[1]
        return new_values

    def __get_saved_value_count(self, file):
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

def normalize_file(file, temporary_path, normalization_values):
    fileName = file.split('/')[-1]
    data = pd.read_csv(file)
    if 'Unnamed: 0' in data.columns:
        data = data.drop(columns=['Unnamed: 0'])
    if 'labevents_Unnamed: 0' in data.columns:
        data = data.drop(columns=['labevents_Unnamed: 0'])
    data = normalize_dataframe(data, normalization_values)
    data.to_csv(temporary_path + fileName, index=False)
    return file, temporary_path + fileName

def normalize_dataframe(data, normalization_values):
    """
    Normalize data using the normalization_values (max and min for the column)
    :param data: the data to be normalized
    :return: the data normalized
    """
    for column in data.columns:
        data.loc[:, column] = z_score_normalization(column, data[column], normalization_values)
    return data

def min_max_normalization(column, series, normalization_values):
    max = normalization_values[column]['max']
    min = normalization_values[column]['min']
    return series.apply(lambda x: (x - min) / (max - min))

def z_score_normalization(column, series, normalization_values):
    # If std is equal to 0, all columns have the same value
    if normalization_values[column]['std'] != 0:
        mean = normalization_values[column]['mean']
        std = normalization_values[column]['std']
        return series.apply(lambda x: (x - mean) / std)
    return series

class Normalization(object):

    def __init__(self, normalization_values, temporary_path='./data_tmp/'):
        self.normalization_values = normalization_values
        self.temporary_path = temporary_path
        if not os.path.exists(temporary_path):
            os.mkdir(temporary_path)
        if not temporary_path.endswith('/'):
            temporary_path += '/'
        self.normalize_file = partial(normalize_file, normalization_values=normalization_values,
                                      temporary_path=temporary_path)
        self.new_paths = None

    def normalize_files(self, filesList):
        """
        Normalize all files in a list of paths
        :param filesList: the list of files path
        :return: a new list for the paths of the normalized data
        """
        with mp.Pool(processes=6) as pool:
            result_pairs = pool.map(self.normalize_file, filesList)
        self.new_paths = dict()
        for pair in result_pairs:
            self.new_paths[pair[0]] = pair[1]

    def get_new_paths(self, files_list):
        if self.new_paths is not None:
            new_list = []
            for file in files_list:
                new_list.append(self.new_paths[file])
            return new_list
        else:
            raise Exception("Data not normalized!")
