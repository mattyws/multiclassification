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

import functions

def escape_invalid_xml_characters(text):
    text = escape(text)
    text = quoteattr(text)
    text = "".join(ch for ch in text if unicodedata.category(ch)[0]!="C")
    return text


def escape_html_special_entities(text):
    return html.unescape(text)


def split_data_for_ctakes(icustayids, noteevents_path=None, ctakes_data_path=None, manager_queue=None):
    tokenizer = WhitespaceTokenizer()
    for icustay in icustayids:
        if manager_queue is not None:
            manager_queue.put(icustay)
        if not os.path.exists(noteevents_path + "{}.csv".format(icustay)) \
                or os.path.exists(ctakes_data_path + str(icustay) + '/'):
            continue
        icustay_path = ctakes_data_path + str(icustay) + '/'
        if not os.path.exists(icustay_path):
            os.mkdir(icustay_path)
        noteevents = pandas.read_csv(noteevents_path + "{}.csv".format(icustay))
        for index, note in noteevents.iterrows():
            new_filename = "{}_{}".format(index, note['Unnamed: 0'])
            with open(icustay_path + new_filename, 'w') as file:
                text = escape_invalid_xml_characters(note['Note'])
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
            print(child.attrib)
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


def merge_ctakes_result_to_csv(icustayids, texts_path=None, ctakes_result_path=None,
                               sentences_data_path=None, merged_results_path=None, manager_queue=None):
    # TODO: only extract the words that have a medical concept associated with it
    sentence_detector = nltk.data.load('tokenizers/punkt/english.pickle')
    tokenizer = WhitespaceTokenizer()
    for icustay in icustayids:
        if manager_queue is not None:
            manager_queue.put(icustay)
        if not os.path.exists(ctakes_result_path + "{}/".format(icustay)) \
                or os.path.exists(merged_results_path + '{}.csv'.format(icustay)):
            continue
        icustay_xmi_path = ctakes_result_path + str(icustay) + '/'
        icustay_text_path = texts_path + str(icustay) + '/'
        xmls = [icustay_xmi_path + x for x in os.listdir(icustay_xmi_path)]
        xmls.sort()
        texts = [icustay_text_path + x for x in os.listdir(icustay_text_path)]
        texts.sort()
        icu_cuis = []
        icustay_sentences = []
        for xml, text in zip(xmls, texts):
            text_cuis = dict()
            text_cuis['timestamp'] = text.split('/')[-1].split('_')[1]
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
                print(word)
                text_cuis['words'].append(word)
            icu_cuis.append(text_cuis)
            print(text)


            # for sentence in sentence_detector.tokenize(text):
            #     # sentence = sentence.strip()
            #     begin = text.index(sentence)
            #     end = begin + len(sentence)
            #     # end += len(sentence)
            #     words_references = get_references_from_sentence(words, sentence, begin, end)
            #     #Creating a copy just to not change the original reference
            #     words_references = copy.deepcopy(words_references)
            #     for reference in words_references:
            #         reference['begin'] -= begin
            #         reference['end'] -= begin
            #     # Look for multi word expressions
            #     multiwords_references, already_added_references = get_multiwords_references(words_references)
            #     # Getting words that were not multiwords or part of it
            #     not_multiwords = []
            #     for word_reference in words_references:
            #         is_not_multiword = True
            #         for added_reference in already_added_references:
            #             if added_reference['word'] == word_reference['word'] \
            #                 and added_reference['begin'] == word_reference['begin']:
            #                 is_not_multiword = False
            #         if is_not_multiword:
            #             not_multiwords.append(copy.deepcopy(word_reference))
            #     for reference in list(itertools.product(*multiwords_references)):
            #         reference = list(reference)
            #         reference.extend(not_multiwords)
            #         # First, copy the object with each of its CUI, if it have more than one CUI
            #         new_reference = []
            #         for item in reference:
            #             cui_object_list = []
            #             for cui in item['cuis']:
            #                 cui_object = copy.deepcopy(item)
            #                 cui_object['cui'] = cui
            #                 cui_object_list.append(cui_object)
            #             new_reference.append(cui_object_list)
            #         for item in list(itertools.product(*new_reference)):
            #             copied_reference = copy.deepcopy(item)
            #             new_sentence = copy.copy(sentence)
            #             for index in range(len(copied_reference)):
            #                 sentence_len = len(new_sentence)
            #                 new_sentence = new_sentence[0:copied_reference[index]['begin']] \
            #                                + copied_reference[index]['cui'] \
            #                                + new_sentence[copied_reference[index]['end']:len(new_sentence)]
            #                 len_diff = len(new_sentence) - sentence_len
            #                 for item2 in copied_reference:
            #                     if item2['begin'] > copied_reference[index]['begin']:
            #                         item2['begin'] += len_diff
            #                         item2['end'] += len_diff
            #             text_sentences.append(new_sentence)
            #     # Now replace the CUIs in text and duplicate the sentence if is the case
            # icustay_sentences.extend(text_sentences)
            # icu_cuis.append(text_cuis)
        for text_cuis in icu_cuis:
            # text_cuis['cuis'] = sorted(text_cuis['cuis'], key=lambda i: i['begin'])
            cuis = []
            for attrib in text_cuis['cuis']:
                for cui in attrib['cuis']:
                    cuis.append(cui)
            text_cuis['cuis'] = cuis
        icu_cuis = pandas.DataFrame(icu_cuis)
        icu_cuis['timestamp'] = pandas.to_datetime(icu_cuis['timestamp'], format=parameters['datetime_pattern'])
        icu_cuis = icu_cuis.sort_values(by=['timestamp'])
        print(icu_cuis)
        exit()
        icu_cuis.to_csv(merged_results_path + '{}.csv'.format(icustay), index=False)
        # with open(sentences_data_path + '{}.txt'.format(icustay), 'w') as file:
        #     for sentence in icustay_sentences:
        #         file.write(sentence + '\n')

parameters = functions.load_parameters_file()

dataset_csv = pandas.read_csv(parameters['mimic_data_path'] + parameters['dataset_file_name'])
noteevents_path = parameters['mimic_data_path'] + parameters['noteevents_anonymized_tokens_normalized']
icustays = dataset_csv['icustay_id']
icustays = np.array_split(icustays, 10)
ctakes_data_path = parameters['mimic_data_path'] + parameters['ctakes_data_path']
ctakes_result_data_path = parameters['mimic_data_path'] + parameters['ctakes_output_path']
uids_data_path = parameters['mimic_data_path'] + parameters['noteevents_ctakes_processed_data_path']
sentences_data_path = parameters['mimic_data_path'] + parameters['noteevents_cuis_normalized_sentences']
if not os.path.exists(ctakes_data_path):
    os.mkdir(ctakes_data_path)
if not os.path.exists(ctakes_result_data_path):
    os.mkdir(ctakes_result_data_path)
if not os.path.exists(uids_data_path):
    os.mkdir(uids_data_path)
if not os.path.exists(sentences_data_path):
    os.mkdir(sentences_data_path)

with mp.Pool(processes=4) as pool:
    m = mp.Manager()
    queue = m.Queue()
    partial_split_data_ctakes = partial(split_data_for_ctakes,
                                        noteevents_path = noteevents_path,
                                        ctakes_data_path=ctakes_data_path,
                                        manager_queue=queue)
    print("===== Spliting events into different files =====")
    map_obj = pool.map_async(partial_split_data_ctakes, icustays)
    consumed = 0
    while not map_obj.ready():
        for _ in range(queue.qsize()):
            queue.get()
            consumed += 1
        sys.stderr.write('\rdone {0:%}'.format(consumed / len(dataset_csv)))
    if queue.qsize() != 0:
        for _ in range(queue.qsize()):
            queue.get()
            consumed += 1
        sys.stderr.write('\rdone {0:%}'.format(consumed / len(dataset_csv)))
    ctakes_params = functions.load_ctakes_parameters_file()
    dirname = os.path.dirname(os.path.realpath(__file__)) + '/'
    # ctakes_command = "sh {}bin/runClinicalPipeline.sh  -i {}  --xmiOut {}  --user {}  --pass {}"\
    #     .format(ctakes_params['ctakes_path'], dirname + ctakes_data_path, dirname + ctakes_result_data_path,
    #             ctakes_params['umls_username'], ctakes_params['umls_password'])
    # process = subprocess.Popen(ctakes_command, shell=True, stdout=subprocess.PIPE)
    # for line in process.stdout:
    #     print(line)
    # process.wait()
    partial_merge_results = partial(merge_ctakes_result_to_csv, texts_path=ctakes_data_path,
                                    ctakes_result_path=ctakes_result_data_path,
                                    sentences_data_path=sentences_data_path,
                                    merged_results_path=uids_data_path, manager_queue=queue)
    partial_merge_results(icustays[0])
    exit()
    print("===== Merging events into a csv =====")
    map_obj = pool.map_async(partial_merge_results, icustays)
    consumed = 0
    while not map_obj.ready():
        for _ in range(queue.qsize()):
            queue.get()
            consumed += 1
        sys.stderr.write('\rdone {0:%}'.format(consumed / len(dataset_csv)))