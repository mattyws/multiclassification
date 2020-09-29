import html
import json
import logging
import os
import pickle
import sys
from datetime import datetime, timedelta
import time
import re
from itertools import islice
from xml.sax.saxutils import escape, quoteattr

import nltk
import pandas as pd

from os.path import exists, join, abspath
from os import pathsep

import unicodedata

from nltk import WhitespaceTokenizer

from adapter import Word2VecTrainer, Doc2VecTrainer
from resources.data_generators import NoteeventsTextDataGenerator, TaggedNoteeventsDataGenerator

DATE_PATTERN = "%Y-%m-%d"
DATETIME_PATTERN = "%Y-%m-%d %H:%M:%S"


def load_parameters_file():
    if not os.path.exists('../parameters.json'):
        raise FileNotFoundError("Parameter file doesn't exists!")
    parameters = json.load(open('../parameters.json'))
    return parameters

def chunk_lst(data, SIZE=10000):
    it = iter(data)
    for i in range(0, len(data), SIZE):
        yield [k for k in islice(it, SIZE)]


def train_representation_model(files_paths, saved_model_path, min_count, size, workers, window, iterations, noteevents_iterator=None,
                               preprocessing_pipeline=None, word2vec=True, hs=1, dm=1, negative=0):
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    model_trainer = None
    if word2vec and noteevents_iterator is None:
        noteevents_iterator = NoteeventsTextDataGenerator(files_paths, preprocessing_pipeline=preprocessing_pipeline)
        model_trainer = Word2VecTrainer(min_count=min_count, size=size, workers=workers, window=window, iter=iterations)
    elif not word2vec and noteevents_iterator is None:
        noteevents_iterator = TaggedNoteeventsDataGenerator(files_paths, preprocessing_pipeline=preprocessing_pipeline)
        model_trainer = Doc2VecTrainer(min_count=min_count, size=size, workers=workers, window=window, iter=iterations,
                                       hs=hs, dm=dm, negative=negative)
    if model_trainer is not None and os.path.exists(saved_model_path):
        model = model_trainer.load_model(saved_model_path)
        return model
    elif model_trainer is not None:
        model_trainer.train(noteevents_iterator)
        model_trainer.save(saved_model_path)
        return model_trainer.model

def filter_events_before_infection(events, admittime, infection_time, preceding_time,
                                   datetime_pattern=DATETIME_PATTERN, time_key="charttime"):
    """
    Get events that occur from admission time until infection time minus preceding time
    :param events: the events
    :param admittime: the admission time
    :param infection_time: the infection time
    :param preceding_time: the preceding time to get the events
    :param datetime_pattern: the pattern used to store time
    :param key: the dictionary key that has the event time
    :return: 
    """
    admittime_datetime = datetime.strptime(admittime, datetime_pattern)
    infection_datetime = datetime.strptime(infection_time, datetime_pattern) - timedelta(hours=preceding_time)
    new_events = []
    for event in events:
        # Pega a data do evento e o transforma em datetime
        event_datetime = datetime.strptime(event[time_key], datetime_pattern)
        # Compara se o evento aconteceu entre a data de adimissão e a data de infecção (já alterada)
        if event_datetime > admittime_datetime and event_datetime <= infection_datetime:
            new_events.append(event)
    return new_events



def filter_since_time(events_object, time_str, max_interval, datetime_pattern=DATETIME_PATTERN, key="charttime", after=False):
    time_point = time.strptime(time_str, datetime_pattern)
    filtered_objects = []
    for event in events_object:
        if len(event[key]) > 0:
            event_date = time.strptime(event[key], datetime_pattern)
            if after:
                difference = (time.mktime(event_date) - time.mktime(time_point)) / 3600
            else:
                difference = (time.mktime(time_point) - time.mktime(event_date)) / 3600
            if difference >= 0 and difference <= max_interval:
                filtered_objects.append(event)
    return filtered_objects




def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]


def search_file(filename, search_path):
    """Given a search path, find file
    """
    file_path = None
    for path in search_path:
        if exists(join(path, filename)):
            file_path = path
            break
    if file_path:
        return abspath(join(file_path, filename))
    return None

def get_file_path_from_id(id, search_paths):
    file_path = search_file('{}.json'.format(id), search_paths)
    return file_path

def get_files_by_ids(ids, search_paths):
    """
    Get the path to the json files from a list of ids
    :param ids: the ids to search for file
    :param search_paths: the paths where the json files are listed
    :return: a dictionary of id : path to json file or None in case of no file found
    """
    files_paths = dict()
    for id in ids:
        path = get_file_path_from_id(id, search_paths)
        files_paths[id] = path
    return files_paths


def chartevents_is_error(event):
    return (pd.notnull(event['ERROR']) and event['ERROR'] != 0) \
           or (pd.notnull(event['STOPPED']) and event['STOPPED'] != 'NotStopd')
           # or (pd.isnull(event['ERROR']) and pd.isnull(event['STOPPED']))

def noteevents_is_error(event):
    return event['ISERROR'] == 1

def is_noteevent_category(event, categories):
    categories = [x.lower() for x in categories]
    if event['CATEGORY'].lower() in categories:
        return True
    return False


def event_is_error(event_label, event, noteevent_category_to_delete=None):
    """
    Check if the event passed as parameter is an error
    :param event_label: the table from where this event is
    :param event: a pandas.Series or similar representing the event
    :return: True if is a error, false otherwise
    """
    if event_label == 'CHARTEVENTS':
        return chartevents_is_error(event)
    elif event_label == 'LABEVENTS':
        # Labevents has no error label
        return False
    elif event_label == 'NOTEEVENTS':
        is_category_to_delete = False
        if noteevent_category_to_delete is not None:
            is_category_to_delete = is_noteevent_category(event, noteevent_category_to_delete)
        return noteevents_is_error(event) or is_category_to_delete
    else:
        raise NotImplemented("Handling error for this table is not implemented yet, exiting.")


def get_event_itemid_and_value(event_label, event):
    """
    Get the value and its id based from which table the event is.
    :param event_label: the table from where this event is
    :param event: a pandas.Series or similar representing the event
    :return:
    """
    if event_label == 'NOTEEVENTS':
        itemid = "Note"
        event_value = event['TEXT']
    elif event_label == 'CHARTEVENTS' or event_label == 'LABEVENTS':
        # Get values and store into a variable, just to read easy and if the labels change
        itemid = event['ITEMID']
        # print(event['VALUE'], event['VALUENUM'])
        if pd.isnull(event['VALUENUM']):
            event_value = str(event['VALUE'])
        else:
            event_value = float(event['VALUENUM'])
    else:
        raise NotImplemented("Event label don't have a filter for its value and itemid!")
    return itemid, event_value


def divide_by_events_lenght(data_list, classes, sizes_filename=None, classes_filename=None):
    """
    Divide a dataset based on their number of timesteps
    :param data_list: list of data
    :param classes: labels for these data
    :param sizes_filename: filename used to save the final sizes object
    :param classes_filename: filename to save the final labels object
    :return:
    """
    sizes = None
    labels = None
    if sizes_filename is not None and classes_filename is not None:
        if os.path.exists(sizes_filename):
            with open(sizes_filename, 'rb') as sizes_handler:
                sizes = pickle.load(sizes_handler)
        if os.path.exists(classes_filename):
            with open(classes_filename, 'rb') as sizes_handler:
                labels = pickle.load(sizes_handler)
    if sizes is None and labels is None:
        sizes = dict()
        labels = dict()
        aux = 0
        for d, c in zip(data_list, classes):
            sys.stderr.write('\rdone {0:%}'.format(aux / len(data_list)))
            aux += 1
            with open(d, 'rb') as file_handler:
                try:
                    values = pickle.load(file_handler)
                except Exception as e:
                    print(d)
                    print("test")
                    print(e)
                    raise ValueError()
                if len(values) not in sizes.keys():
                    sizes[len(values)] = []
                    labels[len(values)] = []
                sizes[len(values)].append(d)
                labels[len(values)].append(c)
        if sizes_filename is not None and classes_filename is not None:
            with open(sizes_filename, 'wb') as sizes_handler:
                pickle.dump(sizes, sizes_handler)
            with open(classes_filename, 'wb') as sizes_handler:
                pickle.dump(labels, sizes_handler)
    return sizes, labels


def mixed_divide_by_events_lenght(data_df:pd.DataFrame, path_column, sizes_filename=None):
    """
    Divide a dataset based on their number of timesteps
    :param data_list: list of data
    :param classes: labels for these data
    :param sizes_filename: filename used to save the final sizes object
    :param classes_filename: filename to save the final labels object
    :return:
    """
    sizes = None
    if sizes_filename is not None:
        if os.path.exists(sizes_filename):
            with open(sizes_filename, 'rb') as sizes_handler:
                sizes = pickle.load(sizes_handler)
    if sizes is None:
        sizes = dict()
        aux = 0
        for index, row in data_df.iterrows():
            sys.stderr.write('\rdone {0:%}'.format(aux / len(data_df)))
            with open(row[path_column], 'rb') as file_handler:
                try:
                    values = pickle.load(file_handler)
                except Exception as e:
                    print(row[path_column])
                    print("test")
                    print(e)
                    raise ValueError()
                if len(values) not in sizes.keys():
                    sizes[len(values)] = []
                sizes[len(values)].append(row['episode'])
            aux += 1
        if sizes_filename is not None:
            with open(sizes_filename, 'wb') as sizes_handler:
                pickle.dump(sizes, sizes_handler)
    return sizes


def load_ctakes_parameters_file():
    if not os.path.exists('ctakes_parameters.json'):
        raise FileNotFoundError("cTakes parameter file doesn't exists!")
    parameters = json.load(open('ctakes_parameters.json'))
    return parameters


def remove_only_special_characters_tokens(tokens):
    new_tokens = []
    for token in tokens:
        if not re.match(r'^[\W_]+$', token):
            new_tokens.append(token)
    return new_tokens


def test_model(kerasAdapter, dataTestGenerator, fold, return_predictions=False):
    evaluation = kerasAdapter.predict_generator(dataTestGenerator)
    metrics = evaluation.metrics
    metrics["fold"] = fold
    if return_predictions:
        result_dict = dict()
        for f, r in zip(evaluation.files, evaluation.predictions_scores):
            result_dict[f] = r
        return metrics, result_dict
    return metrics

def print_with_time(text):
    print("{} ===== {} =====".format(datetime.now().strftime("%d/%m %H:%M:%S"), text))

def remove_sepsis_mentions(tokens):
    sepsis_texts = ['sepsis', 'septic', 'septicemia']
    tokens = [token for token in tokens if token not in sepsis_texts]
    return tokens


def escape_invalid_xml_characters(text):
    text = escape(text)
    text = quoteattr(text)
    text = "".join(ch for ch in text if unicodedata.category(ch)[0] != "C")
    return text


def escape_html_special_entities(text):
    return html.unescape(text)


def text_to_lower(text):
    return text.lower()

def whitespace_tokenize_text(text):
    tokenizer = WhitespaceTokenizer()
    return tokenizer.tokenize(str(text))

def tokenize_text(text):
    sentence_detector = nltk.data.load('tokenizers/punkt/english.pickle')
    return sentence_detector.tokenize(text)

def remove_multiword_token(tokens):
    result_tokens = []
    for token in tokens:
        if ' ' not in token:
            result_tokens.append(token)
    return result_tokens

def tokenize_sentences(sentences):
    tokenized_sentences = []
    for sentence in sentences:
        tokenized_sentences.append(whitespace_tokenize_text(sentence))
    return tokenized_sentences


def get_ensemble_results(meta_model_results, weak_models_results, ensemble_parameters):
    pass

def get_antibiotic_prescriptions(prescriptions):
    antibiotics_names = ['adoxa', 'ala-tet', 'alodox', 'amikacin', 'amikin', 'amoxicillin',
                         'clavulanate', 'ampicillin', 'augmentin', 'avelox', 'avidoxy', 'azactam', 'azithromycin',
                         'aztreonam', 'axetil', 'bactocill', 'bactrim', 'bethkis', 'biaxin', 'bicillin l-a', 'cayston',
                         'cefazolin', 'cedax', 'cefoxitin', 'ceftazidime', 'cefaclor', 'cefadroxil', 'cefdinir', 'cefditoren',
                         'cefepime', 'cefotetan', 'cefotaxime', 'cefpodoxime', 'cefprozil', 'ceftibuten', 'ceftin',
                         'cefuroxime ', 'cefuroxime', 'cephalexin', 'chloramphenicol', 'cipro', 'ciprofloxacin', 'claforan',
                         'clarithromycin', 'cleocin', 'clindamycin', 'cubicin', 'dicloxacillin', 'doryx', 'doxycycline',
                         'duricef', 'ery-tab', 'eryped', 'eryc', 'erythrocin', 'erythromycin', 'factive', 'flagyl',
                         'fortaz', 'furadantin', 'garamycin', 'gentamicin', 'kanamycin', 'keflex', 'ketek', 'levaquin',
                         'levofloxacin', 'lincocin', 'macrobid', 'macrodantin', 'maxipime', 'mefoxin', 'metronidazole',
                         'minocin', 'minocycline', 'monodox', 'morgidox', 'moxatag', 'moxifloxacin', 'myrac',
                         'nafcillin sodium', 'nicazel doxy 30', 'nitrofurantoin', 'noroxin', 'ocudox', 'ofloxacin',
                         'omnicef', 'oracea', 'oraxyl', 'oxacillin', 'pc pen vk', 'pce dispertab', 'panixine',
                         'pediazole', 'penicillin', 'periostat', 'pfizerpen', 'piperacillin', 'tazobactam', 'primsol',
                         'proquin', 'raniclor', 'rifadin', 'rifampin', 'rocephin', 'smz-tmp', 'septra', 'septra ds',
                         'septra', 'solodyn', 'spectracef', 'streptomycin sulfate', 'sulfadiazine', 'sulfamethoxazole',
                         'trimethoprim', 'sulfatrim', 'sulfisoxazole', 'suprax', 'synercid', 'tazicef', 'tetracycline',
                         'timentin', 'tobi', 'tobramycin', 'trimethoprim', 'unasyn', 'vancocin', 'vancomycin', 'vantin',
                         'vibativ', 'vibra-tabs', 'vibramycin', 'zinacef', 'zithromax', 'zmax', 'zosyn', 'zyvox']
    drug_types = ['MAIN', 'ADDITIVE']
    route_eq = ['ou', 'os', 'od', 'au', 'as', 'ad', 'tp']
    route_in = ['eye', 'cream', 'desensitization', 'ophth oint', 'gel']
    prescriptions.loc[:, 'DRUG'] = prescriptions['DRUG'].astype('string').str.lower()
    prescriptions.loc[:, 'ROUTE'] = prescriptions['ROUTE'].astype('string').str.lower()
    prescriptions = prescriptions[prescriptions['DRUG_TYPE'].isin(drug_types)]
    antibiotics_prescriptions = pd.DataFrame([])
    for index, prescription in prescriptions.iterrows():
        if pd.notna(prescription['ROUTE']):
            if prescription['ROUTE'] not in route_eq:
                if any([prescription['ROUTE'] in x for x in route_in]):
                    continue
                if any([prescription['DRUG'] in x for x in antibiotics_names]):
                    antibiotics_prescriptions = antibiotics_prescriptions.append(prescription)
    return antibiotics_prescriptions

def remove_columns_for_classification(data, columns=None):
    if columns is None:
        columns = ['icustay_id', 'starttime', 'charttime', 'endtime', 'bucket']
    columns.extend([x for x in data.columns if 'Unnamed' in x])
    for column in columns:
        if column in data.columns:
            data = data.drop(columns=[column])
    return data
