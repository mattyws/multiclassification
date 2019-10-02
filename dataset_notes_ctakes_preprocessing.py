import os
import subprocess
from functools import partial
from xml.sax.saxutils import escape, quoteattr, unescape
import xml.etree.ElementTree as ET
import nltk.data

import pandas
import html
import unicodedata
import multiprocessing as mp
import numpy as np
import sys

from nltk import WhitespaceTokenizer

import functions

def escape_invalid_xml_characters(text):
    text = escape(text)
    text = quoteattr(text)
    text = "".join(ch for ch in text if unicodedata.category(ch)[0]!="C")
    return text

def split_data_for_ctakes(icustayids, noteevents_path=None, ctakes_data_path=None, manager_queue=None):
    for icustay in icustayids:
        if manager_queue is not None:
            manager_queue.put(icustay)
        if not os.path.exists(noteevents_path + "{}.csv".format(icustay)):
            continue
        icustay_path = ctakes_data_path + str(icustay) + '/'
        if not os.path.exists(icustay_path):
            os.mkdir(icustay_path)
        noteevents = pandas.read_csv(noteevents_path + "{}.csv".format(icustay))
        for index, note in noteevents.iterrows():
            new_filename = "{}_{}".format(index, note['Unnamed: 0'])
            with open(icustay_path + new_filename, 'w') as file:
                file.write(escape_invalid_xml_characters(note['Note']))

def merge_ctakes_result_to_csv(icustayids, texts_path=None, ctakes_result_path=None, merged_results_path=None, manager_queue=None):
    #TODO: save two files - one csv with CUI's for each text, and other with the tokenized sentences for the word2vec
    sentence_detector = nltk.data.load('tokenizers/punkt/english.pickle')
    tokenizer = WhitespaceTokenizer()
    for icustay in icustayids:
        if manager_queue is not None:
            manager_queue.put(icustay)
        if not os.path.exists(ctakes_result_path + "{}/".format(icustay)):
            print("continua")
            continue
        icustay_xmi_path = ctakes_result_path + str(icustay) + '/'
        icustay_text_path = texts_path + str(icustay) + '/'
        xmls = [icustay_xmi_path + x for x in os.listdir(icustay_xmi_path)]
        texts = [icustay_text_path + x for x in os.listdir(icustay_text_path)]
        data = []
        for xml, text in zip(xmls, texts):
            noteevent = dict()
            words = dict()
            print(xml, text)
            # Get the original text, we could got it from the xml result file,
            # but I choose not to just to not make a operation on the xml
            with open(text) as text_file:
                text = text_file.read()
            tree = ET.parse(xml)
            root = tree.getroot()
            # Getting the words that reference a medical concept at the text, and its CUI
            for child in root.iter('*'):
                # The words are marked with the textsem tag and the medical procedures, medication etc have Mention
                # in their names
                if '{http:///org/apache/ctakes/typesystem/type/textsem.ecore}' in child.tag \
                        and "Mention" in child.tag:
                    print("--------------------------------------------------")
                    # Get the word marked by this tag
                    word = text[int(child.attrib['begin']):int(child.attrib['end'])].lower()
                    if word not in words.keys():
                        words[word] = []
                    word_attrib = dict()
                    word_attrib['begin'] = int(child.attrib['begin'])
                    word_attrib['end'] = int(child.attrib['end'])
                    word_attrib['word'] = word
                    word_attrib['cuis'] = set()
                    print(word)
                    print(child.tag, child.attrib)
                    # Now go after their CUIs and add it to the set at the words dictionary
                    for ontology in child.attrib['ontologyConceptArr'].split(' '):
                        umls_ref = root.find(
                            '{http:///org/apache/ctakes/typesystem/type/refsem.ecore}UmlsConcept[@{http://www.omg.org/XMI}id="'
                            + ontology + '"]')
                        word_attrib['cuis'].add(umls_ref.attrib['cui'])
                        print(umls_ref.attrib)
                    print("--------------------------------------------------")
                    words[word].append(word_attrib)
            # print(words)
            print(text)
            begin = 0
            end = 0
            for sentence in sentence_detector.tokenize(text.strip().lower()):
                sentence = sentence.strip()
                print("=====")
                end += len(sentence)
                sentence_references = []
                for word in words.keys():
                    if word in sentence:
                        for word_attrib in words[word]:
                            if word_attrib['begin'] >= begin and word_attrib['begin'] <= end:
                                sentence_references.append(word_attrib)
                print(html.unescape(sentence).replace('\n', ' '))
                print(begin, end, len(sentence))
                print(sentence_references)
                print("=====")
                begin = end
            exit()
        # data = pandas.DataFrame(data)
        # data.to_csv(merged_results_path + "{}.csv".format(icustay), index=False)

parameters = functions.load_parameters_file()

dataset_csv = pandas.read_csv(parameters['mimic_data_path'] + parameters['dataset_file_name'])
noteevents_path = parameters['mimic_data_path'] + parameters['noteevents_anonymized_tokens_normalized']
icustays = dataset_csv['icustay_id']
icustays = np.array_split(icustays, 10)
ctakes_data_path = parameters['mimic_data_path'] + parameters['ctakes_data_path']
ctakes_result_data_path = parameters['mimic_data_path'] + parameters['ctakes_output_path']
uids_data_path = parameters['mimic_data_path'] + parameters['noteevents_ctakes_processed_data_path']
if not os.path.exists(ctakes_data_path):
    os.mkdir(ctakes_data_path)
if not os.path.exists(ctakes_result_data_path):
    os.mkdir(ctakes_result_data_path)
if not os.path.exists(uids_data_path):
    os.mkdir(uids_data_path)

with mp.Pool(processes=4) as pool:
    m = mp.Manager()
    queue = m.Queue()
    # partial_split_data_ctakes = partial(split_data_for_ctakes,
    #                                     noteevents_path = noteevents_path,
    #                                     ctakes_data_path=ctakes_data_path,
    #                                     manager_queue=queue)
    # # TODO : process the fales and put it into a format that can be readable for the ctakes (ID_TIME.txt)
    # print("===== Spliting events into different files =====")
    # map_obj = pool.map_async(partial_split_data_ctakes, icustays)
    # consumed = 0
    # while not map_obj.ready():
    #     for _ in range(queue.qsize()):
    #         queue.get()
    #         consumed += 1
    #     sys.stderr.write('\rdone {0:%}'.format(consumed / len(dataset_csv)))
    #
    # # TODO : execute the command line for the ctakes pipeline
    # ctakes_params = functions.load_ctakes_parameters_file()
    # dirname = os.path.dirname(os.path.realpath(__file__)) + '/'
    # ctakes_command = "sh {}bin/runClinicalPipeline.sh  -i {}  --xmiOut {}  --user {}  --pass {}"\
    #     .format(ctakes_params['ctakes_path'], dirname + ctakes_data_path, dirname + ctakes_result_data_path,
    #             ctakes_params['umls_username'], ctakes_params['umls_password'])
    # print(ctakes_command)
    # process = subprocess.Popen(ctakes_command, shell=True, stdout=subprocess.PIPE)
    # for line in process.stdout:
    #     print(line)
    # process.wait()
    # TODO : merge the files for the same id into a csv with the results for the ctakes
    partial_merge_results = partial(merge_ctakes_result_to_csv, texts_path=ctakes_data_path,
                                    ctakes_result_path=ctakes_result_data_path,
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