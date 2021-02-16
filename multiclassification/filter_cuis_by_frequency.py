import copy
import itertools
import os
import subprocess
from functools import partial
from html.parser import HTMLParser
from xml.sax.saxutils import escape, quoteattr, unescape
import xml.etree.ElementTree as ET
import nltk.data
from ast import literal_eval
import pandas as pd
import html
import unicodedata
import multiprocessing as mp
import numpy as np
import sys

from nltk import WhitespaceTokenizer, RegexpTokenizer

from resources import functions
import multiclassification.constants as constants

from multiclassification.parameters.dataset_parameters import parameters

def count_frequencies(paths:list)->pd.Series:
    cuis_frequencies = dict()
    for path in paths:
        cuis = pd.read_csv(path)
        columns = cuis.columns
        for column in columns:
            if column == "starttime" or column == "endtime" or column == "bucket":
                continue
            if column not in cuis_frequencies.keys():
                cuis_frequencies[column] = 0
            cuis_frequencies[column] += 1
    cuis_frequencies = pd.Series(cuis_frequencies)
    cuis_frequencies = cuis_frequencies.sort_values(ascending=False)
    return cuis_frequencies

def filter_most_frequent_cuis(dataset:pd.DataFrame, frequent_cuis:list):
    return dataset[frequent_cuis]


multiclassification_base_path = os.path.join(parameters['mimic_data_path'], parameters['multiclassification_directory'])

problem = 'mortality'
problem_base_dir = os.path.join(multiclassification_base_path, parameters['{}_directory'.format(problem)])
dataset_path = os.path.join(problem_base_dir, parameters['{}_dataset_csv'.format(problem)])
dataset_csv = pd.read_csv(dataset_path)
print(len(dataset_csv))
dataset = np.array_split(dataset_csv, 10)
bag_of_cuis_files_path = os.path.join(problem_base_dir, parameters['bag_of_cuis_files_path'])
bag_of_cuis_frequent_path = os.path.join(problem_base_dir, parameters['bag_of_cuis_frequent_path'])
if not os.path.exists(bag_of_cuis_frequent_path):
    os.makedirs(bag_of_cuis_frequent_path)


