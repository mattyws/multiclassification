import csv
from functools import partial

import math

import pandas as pd
import numpy as np
import multiprocessing as mp


def get_file_value_counts(file):
    df = pd.read_csv(file)
    if 'Unnamed: 0' in df.columns:
        df = df.set_index(['Unnamed: 0'])
    if 'labevents_Unnamed: 0' in df.columns:
        df = df.drop(columns=['labevents_Unnamed: 0'])
    counts = dict()
    for column in df.columns:
        counts[column] = df[column].value_counts()
        counts[column].index = counts[column].index.map(float)
    return file, counts

class NormalizationValues(object):
    def __init__(self, files_list):
        self.files_list = files_list
        self.get_file_value_counts = get_file_value_counts

    def prepare(self, result_fname=None):
        """
        Prepare the data getting their value counts for each column
        :return:
        """
        # TODO : pickle the count object and load it if alread exists
        if result_fname is None:
            with mp.Pool(processes=6) as pool:
                results = pool.map(self.get_file_value_counts, self.files_list)
                self.counts = dict()
                for result in results:
                    self.counts[result[0]] = result[1]

    def get_normalization_values(self, training_files):
        """
        Get the max, min, mean and std value for each column from a set of csv files used for training the model
        :return: a dict with the values for each column
        """
        values = None
        # Loop each file in dataset
        for file in training_files:
            if values is None:
                values = self.counts[file]
            else:
                for key in values.keys():
                    values[key] = values[key].combine(self.counts[file][key], func = (lambda x1, x2: x1 + x2),
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
    return temporary_path + fileName

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
        self.normalize_file = partial(normalize_file, normalization_values=normalization_values,
                                      temporary_path=temporary_path)

    def normalize_files(self, filesList):
        """
        Normalize all files in a list of paths
        :param filesList: the list of files path
        :return: a new list for the paths of the normalized data
        """
        with mp.Pool(processes=6) as pool:
            newList = pool.map(self.normalize_file, filesList)
        return newList
        # newList = []
        # for file in filesList:
        #     fileName = file.split('/')[-1]
        #     data = pd.read_csv(file)
        #     if 'Unnamed: 0' in data.columns:
        #         data = data.drop(columns=['Unnamed: 0'])
        #     data = self.__normalize(data)
        #     # Sort columns
        #     columns = list(data.columns)
        #     data.to_csv(self.temporary_path+fileName, index=False)
        #     newList.append(self.temporary_path+fileName)
        # return newList


    def __normalize(self, data):
        """
        Normalize data using the normalization_values (max and min for the column)
        :param data: the data to be normalized
        :return: the data normalized
        """
        for column in data.columns:
            data.loc[:, column] = self.__z_score_normalization(column, data[column])
        return data

    def __min_max_normalization(self, column, series):
        max = self.normalization_values[column]['max']
        min = self.normalization_values[column]['min']
        return series.apply(lambda x: (x - min) / (max - min))

    def __z_score_normalization(self, column, series):
        # If std is equal to 0, all columns have the same value
        if self.normalization_values[column]['std'] != 0:
            mean = self.normalization_values[column]['mean']
            std = self.normalization_values[column]['std']
            return series.apply(lambda x: (x - mean) / std)
        return series

    @staticmethod
    def get_normalization_values(filesList):
        """
        Get the max and min value for each column from a set of csv files
        :param filesList: the list of files to get the value
        :return: a dict with the max and min value for each column
        """
        values = dict()
        # Loop each file in dataset
        for file in filesList:
            df = pd.read_csv(file)
            if 'Unnamed: 0' in df.columns:
                df = df.drop(columns=['Unnamed: 0'])
            # Loop each column in file
            for column in df.columns:
                # Add if column don't exist at keys
                if column not in values.keys():
                    values[column] = dict()
                    values[column]['values'] = pd.Series([])
                values[column]['values'] = values[column]['values'].append(df[column])
        for key in values.keys():
            values[key]['max'] = values[key]['values'].max()
            values[key]['min'] = values[key]['values'].min()
            values[key]['mean'] = values[key]['values'].mean()
            values[key]['std'] = values[key]['values'].std()
            values[key]['values'] = None
        return values
