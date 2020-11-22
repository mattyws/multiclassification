import copy
import itertools
import os
import subprocess
from functools import partial
from html.parser import HTMLParser
from xml.sax.saxutils import escape, quoteattr, unescape
import xml.etree.ElementTree as ET
import nltk.data

import pandas
import html
import unicodedata
import multiprocessing as mp
import numpy as np
import sys

from nltk import WhitespaceTokenizer, RegexpTokenizer

from resources import functions
import multiclassification.constants as constants

def escape_invalid_xml_characters(text):
    text = escape(text)
    text = quoteattr(text)
    text = "".join(ch for ch in text if unicodedata.category(ch)[0]!="C")
    return text


def escape_html_special_entities(text):
    return html.unescape(text)


def split_data_for_ctakes(dataset:pandas.DataFrame, multiclassification_base_path=None, ctakes_data_path=None, manager_queue=None):
    tokenizer = WhitespaceTokenizer()
    consumed = 0
    total_rows = len(dataset)
    for index, row in dataset.iterrows():
        consumed += 1
        if manager_queue is not None:
            manager_queue.put(row['icustay_id'])
        else:
            sys.stderr.write('\rdone {0:%}'.format(consumed / total_rows))
        icustay_path = os.path.join(ctakes_data_path, str(row['icustay_id']) )
        if os.path.exists(icustay_path):
            continue
        else:
            os.makedirs(icustay_path)
        textual_data_path = os.path.join(multiclassification_base_path, row['textual_path'])
        noteevents = pandas.read_csv(textual_data_path)
        for index, note in noteevents.iterrows():
            if note['text'] == constants.NO_TEXT_CONSTANT:
                continue
            new_filename = "{}_{}_{}".format(note['bucket'], note['starttime'], note['endtime'])
            with open(os.path.join(icustay_path, new_filename), 'w') as file:
                text = escape_invalid_xml_characters(note['text'])
                text = escape_html_special_entities(text)
                text = functions.remove_only_special_characters_tokens(tokenizer.tokenize(text))
                text = " ".join(text)
                file.write(text)

def get_word_cuis_from_xml(root, text):
    words = dict()
    cuis = []
    # Getting the words that reference a medical concept at the text, and its CUI
    for child in root.iter('*'):
        # The words are marked with the textsem tag and the medical procedures, medication etc have Mention
        # in their names
        if '{http:///org/apache/ctakes/typesystem/type/textsem.ecore}' in child.tag \
                and "Mention" in child.tag:
            # Get the word marked by this tag
            word = text[int(child.attrib['begin']):int(child.attrib['end'])].lower()
            word_attrib = dict()
            word_attrib['begin'] = int(child.attrib['begin'])
            word_attrib['end'] = int(child.attrib['end'])
            word_attrib['word'] = word
            word_attrib['cuis'] = set()
            # Now go after their CUIs and add it to the set at the words dictionary
            for ontology in child.attrib['ontologyConceptArr'].split(' '):
                umls_ref = root.find(
                    '{http:///org/apache/ctakes/typesystem/type/refsem.ecore}UmlsConcept[@{http://www.omg.org/XMI}id="'
                    + ontology + '"]')
                word_attrib['cuis'].add(umls_ref.attrib['cui'])
            word_attrib['cuis'] = list(word_attrib['cuis'])
            if word not in words.keys():
                words[word] = []
                words[word].append(copy.deepcopy(word_attrib))
            else:
                # Check if it is the same word, but with a different xml tag
                is_added = False
                for attrib_added in words[word]:
                    if word_attrib['begin'] == attrib_added['begin']:
                        is_added = True
                        attrib_added['cuis'].extend(word_attrib['cuis'])
                if not is_added:
                    words[word].append(copy.deepcopy(word_attrib))
            cuis.append(copy.deepcopy(word_attrib))
    return words, cuis

def get_references_from_sentence(words, sentence, begin, end):
    words_references = []
    for word in words.keys():
        if word in sentence:
            for word_attrib in words[word]:
                if word_attrib['begin'] >= begin and word_attrib['begin'] <= end:
                    words_references.append(copy.deepcopy(word_attrib))
    return words_references

def get_multiwords_references(words_references):
    multiwords_references = []
    already_added_references = []
    for word_reference in words_references:
        expression_reference = []
        # If it has a space is a multiword expression
        if len(word_reference['word'].split(' ')) > 1:
            # Check if it is not already added, if it is, ignore
            # This is done because multiwords expressions can have a size higher than two
            is_added = False
            for added_reference in already_added_references:
                if added_reference['begin'] == word_reference['begin'] \
                        and added_reference['end'] == word_reference['end']:
                    is_added = True
            if is_added:
                continue
            expression_reference.append(word_reference)
            # Looking if a word in this expression has a CUI of its own
            for word_reference2 in words_references:
                if word_reference2['word'] != word_reference['word'] \
                        and word_reference2['begin'] >= word_reference['begin'] \
                        and word_reference2['end'] <= word_reference['end']:
                    expression_reference.append(copy.deepcopy(word_reference2))
            if len(expression_reference) > 1:
                for reference in expression_reference:
                    already_added_references.append(copy.deepcopy(reference))
                multiwords_references.append(copy.deepcopy(expression_reference))
    return multiwords_references, already_added_references


def merge_ctakes_result_to_csv(dataset:pandas.DataFrame, texts_path=None, ctakes_result_path=None,
                               extracted_words_and_cuis_path=None, manager_queue=None):
    consumed = 0
    returned_paths = []
    for index, row in dataset.iterrows():
        returned_path = dict()
        consumed += 1
        icustay = str(row['icustay_id'])
        extracted_words_and_cuis_icustay_path = os.path.join(extracted_words_and_cuis_path, '{}.csv'.format(icustay))
        icustay_file = dict()
        icustay_file['icustay_id'] = icustay
        icustay_file['ctakes_extracted_words_cuis'] = extracted_words_and_cuis_icustay_path
        if os.path.exists(extracted_words_and_cuis_icustay_path):
            if manager_queue is not None:
                manager_queue.put(icustay_file)
            else:
                sys.stderr.write('\rdone {0:%}'.format(consumed / len(dataset_csv)))
            continue
        icustay_xmi_path = os.path.join(ctakes_result_path, icustay)
        icustay_text_path = os.path.join(texts_path, icustay)
        if not os.path.exists(icustay_xmi_path) or not os.path.exists(icustay_text_path):
            continue
        xmls = [os.path.join(icustay_xmi_path, x) for x in os.listdir(icustay_xmi_path)]
        xmls.sort()
        texts = [os.path.join(icustay_text_path, x) for x in os.listdir(icustay_text_path)]
        texts.sort()
        icu_cuis = []
        for xml, text in zip(xmls, texts):
            text_cuis = dict()
            filename = os.path.basename(text)
            text_cuis['bucket'] = filename.split('_')[0]
            text_cuis['starttime'] = filename.split('_')[1]
            text_cuis['endtime'] = filename.split('_')[2]
            text_sentences = []
            # Get the original text, we could got it from the xml result file,
            # but I choose not to just to not make a operation on the xml
            with open(text) as text_file:
                text = text_file.read()
            tree = ET.parse(xml)
            root = tree.getroot()
            # Getting the words that reference a medical concept at the text, and its CUI
            words, text_cuis['cuis'] = get_word_cuis_from_xml(root, text)
            text_cuis['cuis'] = sorted(text_cuis['cuis'], key=lambda i: i['begin'])
            text = text.lower()
            text_cuis['words'] = []
            for cui in text_cuis['cuis']:
                word = text[cui['begin']:cui['end']]
                text_cuis['words'].append(word)
            icu_cuis.append(text_cuis)
        for text_cuis in icu_cuis:
            # text_cuis['cuis'] = sorted(text_cuis['cuis'], key=lambda i: i['begin'])
            cuis = []
            for attrib in text_cuis['cuis']:
                for cui in attrib['cuis']:
                    cuis.append(cui)
            text_cuis['cuis'] = cuis
        icu_cuis = pandas.DataFrame(icu_cuis)
        icu_cuis['starttime'] = pandas.to_datetime(icu_cuis['starttime'], format=parameters['datetime_pattern'])
        icu_cuis['endtime'] = pandas.to_datetime(icu_cuis['endtime'], format=parameters['datetime_pattern'])
        icu_cuis = icu_cuis.sort_values(by=['bucket'])
        icu_cuis.to_csv(extracted_words_and_cuis_icustay_path, index=False)
        returned_path["ctakes_file"] = icustay_file
        returned_path['icustay'] = int(icustay)
        returned_paths.append(returned_path)
        if manager_queue is not None:
            manager_queue.put(icustay_file)
        else:
            sys.stderr.write('\rdone {0:%}'.format(consumed / len(dataset_csv)))
        # with open(sentences_data_path + '{}.txt'.format(icustay), 'w') as file:
        #     for sentence in icustay_sentences:
        #         file.write(sentence + '\n')
    return returned_paths

from multiclassification.parameters.dataset_parameters import parameters

multiclassification_base_path = os.path.join(parameters['mimic_data_path'], parameters['multiclassification_directory'])

problem = 'mortality'
problem_base_dir = os.path.join(multiclassification_base_path, parameters['{}_directory'.format(problem)])
dataset_path = os.path.join(problem_base_dir, parameters['{}_dataset_csv'.format(problem)])
dataset_csv = pandas.read_csv(dataset_path)
print(len(dataset_csv))
dataset = np.array_split(dataset_csv, 10)
ctakes_data_path = os.path.join(problem_base_dir, parameters['ctakes_input_dir'])
ctakes_result_data_path = os.path.join(problem_base_dir, parameters['ctakes_output_path'])
extracted_words_and_cuis_path = os.path.join(problem_base_dir, parameters['ctakes_processed_data_path'])
if not os.path.exists(ctakes_data_path):
    os.mkdir(ctakes_data_path)
if not os.path.exists(ctakes_result_data_path):
    os.mkdir(ctakes_result_data_path)
if not os.path.exists(extracted_words_and_cuis_path):
    os.mkdir(extracted_words_and_cuis_path)

with mp.Pool(processes=4) as pool:
    m = mp.Manager()
    queue = m.Queue()
    partial_split_data_ctakes = partial(split_data_for_ctakes,
                                        multiclassification_base_path = multiclassification_base_path,
                                        ctakes_data_path=ctakes_data_path,
                                        manager_queue=None)
    print("===== Spliting events into different files =====")
    partial_split_data_ctakes(dataset_csv)
    # map_obj = pool.map_async(partial_split_data_ctakes, dataset)
    # consumed = 0
    # while not map_obj.ready():
    #     for _ in range(queue.qsize()):
    #         queue.get()
    #         consumed += 1
    #     sys.stderr.write('\rdone {0:%}'.format(consumed / len(dataset_csv)))
    # if queue.qsize() != 0:
    #     for _ in range(queue.qsize()):
    #         queue.get()
    #         consumed += 1
    #     sys.stderr.write('\rdone {0:%}'.format(consumed / len(dataset_csv)))
    # ctakes_params = functions.load_ctakes_parameters_file()
    # dirname = os.path.dirname(os.path.realpath(__file__)) + '/'
    # ctakes_command = "sh {}bin/runClinicalPipeline.sh  -i {}  --xmiOut {}  --user {}  --pass {}"\
    #     .format(ctakes_params['ctakes_path'], ctakes_data_path, ctakes_result_data_path,
    #             ctakes_params['umls_username'], ctakes_params['umls_password'])
    # process = subprocess.Popen(ctakes_command, shell=True, stdout=subprocess.PIPE)
    # for line in process.stdout:
    #     print(line)
    # process.wait()
    partial_merge_results = partial(merge_ctakes_result_to_csv, texts_path=ctakes_data_path,
                                    ctakes_result_path=ctakes_result_data_path,
                                    extracted_words_and_cuis_path=extracted_words_and_cuis_path, manager_queue=queue)
    print("===== Merging events into a csv =====")
    icustay_paths = []
    consumed = 0
    partial_merge_results(dataset_csv)
    # map_obj = pool.map_async(partial_merge_results, dataset)
    # while not map_obj.ready() or queue.qsize() != 0:
    #     for _ in range(queue.qsize()):
    #         icustay_file = queue.get()
    #         icustay_paths.append(icustay_file)
    #         consumed += 1
    #     sys.stderr.write('\rdone {0:%}'.format(consumed / len(dataset_csv)))
    # if len(icustay_paths) != 0:
    #     icustay_paths = pandas.DataFrame(icustay_paths)
    #     icustay_paths.to_csv(os.path.join(multiclassification_base_path, 'ctakes_paths.csv'))