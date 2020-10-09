import pandas as pd
import os
import sys

from multiclassification import constants
from multiclassification.parameters.classification_parameters import timeseries_textual_training_parameters as parameters
from resources.data_representation import ClinicalTokenizer
from resources.functions import whitespace_tokenize_text

problem = 'mortality'
problem_path = parameters['multiclassification_base_path'] + parameters[problem+'_directory']
dataset_path = problem_path + parameters[problem+'_dataset_csv']
vocabulary_count_path = problem_path + 'vocabulary.csv'


data_csv = pd.read_csv(dataset_path)
data_csv = data_csv.sort_values(['episode'])
labels = data_csv['label'].unique().tolist()
if not os.path.exists(vocabulary_count_path):
    preprocessing_pipeline = [whitespace_tokenize_text]
    clinical_tokenizer = ClinicalTokenizer()
    tokens = dict()
    consumed = 0
    total_files = len(data_csv)
    for index, row in data_csv.iterrows():
        sys.stderr.write('\rdone {0:%}'.format(consumed / total_files))
        consumed += 1
        path = row['textual_path']
        patient_noteevents = pd.read_csv(path)
        patient_id = os.path.basename(path).split('.')[0]
        for index, text_row in patient_noteevents.iterrows():
            text = text_row['text']
            if text == constants.NO_TEXT_CONSTANT:
                continue
            sentences = clinical_tokenizer.tokenize_sentences(text)
            tokens_in_doc = set()
            for num, sentence in enumerate(sentences):
                processed_sentence = sentence
                if preprocessing_pipeline is not None:
                    for func in preprocessing_pipeline:
                        processed_sentence = func(processed_sentence)

                for token in processed_sentence:
                    if token not in tokens.keys():
                        tokens[token] = dict()
                        tokens[token]['total'] = 0
                        tokens[token]['docs'] = 0
                        for label in labels:
                            tokens[token]['total_label_{}'.format(label)] = 0
                            tokens[token]['unique_docs_{}'.format(label)] = 0
                    if token not in tokens_in_doc:
                        tokens_in_doc.add(token)
                        tokens[token]['docs'] += 1
                        tokens[token]['unique_docs_{}'.format(row['label'])] += 1
                    tokens[token]['total'] += 1
                    tokens[token]['total_label_{}'.format(row['label'])] += 1

    tokens = pd.DataFrame(tokens).transpose()
    tokens.to_csv(vocabulary_count_path, index_label='token')
else:
    tokens = pd.read_csv(vocabulary_count_path)

print()
labels_frequent_words = []
for label in labels:
    fields = ['token', 'unique_docs_{}'.format(label), 'total_label_{}'.format(label)]
    label_vocab = tokens[fields]
    label_vocab = label_vocab.sort_values(by=['total_label_{}'.format(label)], ascending=False).iloc[:200]
    label_frequent_words = set(label_vocab['token'].tolist())
    labels_frequent_words.append(label_frequent_words)
    label_vocab.to_csv(problem_path + 'vocabulary_label_{}.csv'.format(label))

vocab_intersect = labels_frequent_words[0]
for index in range(1, len(labels_frequent_words)):
    vocab_intersect = vocab_intersect.intersection(labels_frequent_words[index])
print(vocab_intersect)
print(len(vocab_intersect))