import os
import pickle
import types
from ast import literal_eval
from functools import partial

import bert
import scispacy
import spacy
import torch
from tensorflow.keras.models import Model

import tensorflow as tf
from tensorflow import keras
from torch.tensor import Tensor
from transformers.modeling_auto import AutoModel
from transformers.tokenization_auto import AutoTokenizer

from biobert import tokenization as biobert_tokenizer

import multiprocessing
# import sentencepiece as spm
import numpy
import numpy as np
import pandas
import sys

from multiclassification import constants

class TransformClinicalTextsRepresentations():
    """
    Changes the representation for patients notes using a word2vec model.
    The patients notes must be into different csv.
    """
    def __init__(self, representation_model, embedding_size=200, text_max_len=None, window=2,
                 representation_save_path=None, is_word2vec=True):
        self.representation_model = representation_model
        self.embedding_size = embedding_size
        self.window = window
        self.text_max_len = text_max_len
        self.representation_save_path = representation_save_path
        if not os.path.exists(representation_save_path):
            os.mkdir(representation_save_path)
        self.new_paths = dict()
        self.lock = None
        self.is_word2vec = is_word2vec
        self.clinical_tokenizer = ClinicalTokenizer()

    def create_embedding_matrix(self, text):
        """
        Transform a tokenized text into a 3 dimensional array with the word2vec model
        :param text: the tokenized text
        :return: the 3 dimensional array representing the content of the tokenized text
        """
        # x = np.zeros(shape=(self.text_max_len, self.embedding_size), dtype='float')
        x = []
        # if len(text) < 3:
        #     return None
        for pos, w in enumerate(text):
            try:
                # x[pos] = self.word2vec_model.wv[w]
                x.append(self.representation_model.wv[w])
            except:
                try:
                    # x[pos] = np.zeros(shape=self.embeddingSize)
                    if pos - self.window < 0:
                        begin = 0
                    else:
                        begin = pos - self.window
                    if pos + self.window > len(text):
                        end = len(text)
                    else:
                        end = pos + self.window
                    word = self.representation_model.predict_output_word(text[begin:end])[0][0]
                    # x[pos] = self.word2vec_model.wv[word]
                    x.append(self.representation_model.wv[word])
                except:
                    # continue
                    # x[pos] = np.zeros(shape=self.embedding_size)
                    x.append(np.zeros(shape=self.embedding_size))
        x = np.array(x)
        return x

    def transform_texts(self, texts_df:pandas.DataFrame, preprocessing_pipeline=[],
                       remove_temporal_axis: bool = False, remove_no_text_constant:bool=False) -> list:
        transformed_texts = []
        sentences_num = 0
        for index, row in texts_df.iterrows():
            note = row['text']
            if remove_no_text_constant and note == constants.NO_TEXT_CONSTANT:
                continue
            sentences = self.clinical_tokenizer.tokenize_sentences(note)
            transformed_sentences = []
            for sentence in sentences:
                sentences_num += 1
                processed_sentence = sentence
                if preprocessing_pipeline is not None:
                    for func in preprocessing_pipeline:
                        processed_sentence = func(processed_sentence)
                if self.is_word2vec:
                    new_representation = self.create_embedding_matrix(processed_sentence)
                else:
                    new_representation = self.representation_model.infer_vector(processed_sentence)
                if new_representation is not None:
                    transformed_sentences.append(new_representation)
                else:
                    print("Is none", note)
                    exit()
            if remove_temporal_axis:
                transformed_texts.extend(transformed_sentences)
            else:
                transformed_sentences = np.mean(transformed_sentences, axis=0)
                transformed_texts.append(transformed_sentences)
        return transformed_texts

    def transform(self, data_df:pandas.DataFrame, text_paths_column:str, preprocessing_pipeline=[],
                       remove_temporal_axis: bool = False, remove_no_text_constant:bool=False) -> pandas.DataFrame:
        episodes = []
        paths = []
        labels = []
        total_files = len(data_df)
        consumed = 0
        for index, row in data_df.iterrows():
            # print(row['episode'])
            sys.stderr.write('\rdone {0:%}'.format(consumed / total_files))
            consumed += 1
            transformed_doc_path = os.path.join(self.representation_save_path, str(row['episode']) + '.pkl')
            if os.path.exists(transformed_doc_path):
                episodes.append(row['episode'])
                paths.append(transformed_doc_path)
                labels.append(row['label'])
                continue
            path = row[text_paths_column]
            data = pandas.read_csv(path)
            transformed_texts = self.transform_texts(data, preprocessing_pipeline=preprocessing_pipeline,
                                                     remove_temporal_axis=remove_temporal_axis,
                                                     remove_no_text_constant=remove_no_text_constant)
            if len(transformed_texts) != 0:
                transformed_texts = numpy.asarray(transformed_texts)
                with open(transformed_doc_path, 'wb') as handler:
                    pickle.dump(transformed_texts, handler)
                episodes.append(row['episode'])
                paths.append(transformed_doc_path)
                labels.append(row['label'])
            else:
                # print(data)
                print("Is empty", path)
                continue
        new_paths = pandas.DataFrame({'path': paths, 'label': labels}, index=episodes)
        return new_paths

    def pad_sequence(self, value, pad_max_len):
        # if len(value) < 3:
        #     return None
        if len(value) >= pad_max_len:
            return value[:pad_max_len]
        else:
            zeros = np.zeros(shape=(pad_max_len, self.embedding_size))
            zeros[: len(value)] = value
            return zeros

    def pad_patient_text(self, doc_paths, pad_max_len=None, pad_data_path=None, manager_queue=None):
        new_paths = dict()
        for path in doc_paths:
            filename = path.split('/')[-1]
            if manager_queue is not None:
                manager_queue.put(path)
            transformed_doc_path = pad_data_path + os.path.splitext(filename)[0] + '.pkl'
            if os.path.exists(transformed_doc_path):
                new_paths[path] = transformed_doc_path
                continue
            with open(path, 'rb') as fhandler:
                data = pickle.load(fhandler)
            padded_data = []
            for value in data:
                padded_value = self.pad_sequence(value, pad_max_len)
                if padded_value is not None:
                    padded_data.append(padded_value)
            if len(padded_data) != 0:
                padded_data = numpy.array(padded_data)
                with open(transformed_doc_path, 'wb') as handler:
                    pickle.dump(padded_data, handler)
                new_paths[path] = transformed_doc_path
        return new_paths

    def pad_new_representation(self, docs_paths, pad_max_len, pad_data_path=None):
        if not os.path.exists(pad_data_path):
            os.mkdir(pad_data_path)
        with multiprocessing.Pool(processes=6) as pool:
            manager = multiprocessing.Manager()
            manager_queue = manager.Queue()
            self.lock = manager.Lock()
            partial_transform_docs = partial(self.pad_patient_text, pad_max_len=pad_max_len, pad_data_path=pad_data_path,
                                             manager_queue=manager_queue)
            # docs_paths = self.new_paths.values()
            # print(docs_paths)
            # exit()
            data = numpy.array_split(docs_paths, 6)
            total_files = len(docs_paths)
            map_obj = pool.map_async(partial_transform_docs, data)
            consumed = 0
            while not map_obj.ready() or manager_queue.qsize() != 0:
                for _ in range(manager_queue.qsize()):
                    manager_queue.get()
                    consumed += 1
                sys.stderr.write('\rdone {0:%}'.format(consumed / total_files))
            print()
            result = map_obj.get()
            padded_paths = dict()
            for r in result:
                padded_paths.update(r)
            self.new_paths = padded_paths


    def get_new_paths(self, files_list):
        if self.new_paths is not None and len(self.new_paths.keys()) != 0:
            new_list = []
            for file in files_list:
                if file in self.new_paths.keys():
                    new_list.append(self.new_paths[file])
            return new_list
        else:
            raise Exception("Data not transformed!")


class TransformClinicalCtakesTextsRepresentations(object):
    """
    Changes the representation for patients notes using a word2vec model.
    The patients notes must be into different csv.
    """
    def __init__(self, representation_model, embedding_size=200, window=2,
                 texts_path=None, representation_save_path=None):
        self.representation_model = representation_model
        self.embedding_size = embedding_size
        self.window = window
        self.texts_path = texts_path
        self.representation_save_path = representation_save_path
        if not os.path.exists(representation_save_path):
            os.mkdir(representation_save_path)
        self.new_paths = dict()
        self.lock = None

    def create_embedding_matrix(self, text):
        """
        Transform a tokenized text into a 3 dimensional array with the word2vec model
        :param text: the tokenized text
        :return: the 3 dimensional array representing the content of the tokenized text
        """
        # x = np.zeros(shape=(self.text_max_len, self.embedding_size), dtype='float')
        x = []
        # if len(text) < 3:
        #     return None
        for pos, w in enumerate(text):
            try:
                # x[pos] = self.word2vec_model.wv[w]
                x.append(self.representation_model.wv[w])
            except:
                try:
                    # x[pos] = np.zeros(shape=self.embeddingSize)
                    if pos - self.window < 0:
                        begin = 0
                    else:
                        begin = pos - self.window
                    if pos + self.window > len(text):
                        end = len(text)
                    else:
                        end = pos + self.window
                    word = self.representation_model.predict_output_word(text[begin:end])[0][0]
                    # x[pos] = self.word2vec_model.wv[word]
                    x.append(self.representation_model.wv[word])
                except:
                    # continue
                    # x[pos] = np.zeros(shape=self.embedding_size)
                    x.append(np.zeros(shape=self.embedding_size))
        x = np.array(x)
        return x

    def transform_docs(self, docs_path, preprocessing_pipeline=[], manager_queue=None):
        new_paths = dict()
        total_files = len(docs_path)
        print(total_files)
        consumed = 0
        for path in docs_path:
            file_name = path.split('/')[-1]
            if manager_queue is None:
                # manager_queue.put(path)
                sys.stderr.write('\rdone {0:%}'.format(consumed / total_files))
                consumed += 1
            else:
                manager_queue.put(path)
            transformed_doc_path = self.representation_save_path + os.path.splitext(file_name)[0] + '.pkl'
            if os.path.exists(transformed_doc_path):
                new_paths[path] = transformed_doc_path
                continue
            data = pandas.read_csv(path)
            transformed_texts = []
            for index, row in data.iterrows():
                try:
                    note = row['words']
                except Exception as e:
                    print(path)
                    raise Exception("deu errado")
                note = literal_eval(note)
                if preprocessing_pipeline is not None:
                    for func in preprocessing_pipeline:
                        note = func(note)
                new_representation = self.create_embedding_matrix(note)
                if new_representation is not None:
                    transformed_texts.extend(new_representation)
            if len(transformed_texts) >= 3:
                transformed_texts = numpy.array(transformed_texts)
                with open(transformed_doc_path, 'wb') as handler:
                    pickle.dump(transformed_texts, handler)
                new_paths[path] = transformed_doc_path
        return new_paths

    def transform(self, docs_paths, preprocessing_pipeline=None):
        self.new_paths = self.transform_docs(docs_paths, preprocessing_pipeline=preprocessing_pipeline)
        # with multiprocessing.Pool(processes=4) as pool:
        #     manager = multiprocessing.Manager()
        #     manager_queue = manager.Queue()
        #     self.lock = manager.Lock()
            # partial_transform_docs = partial(self.transform_docs,
            #                                  preprocessing_pipeline=preprocessing_pipeline,
            #                                  manager_queue=manager_queue)
            # data = numpy.array_split(docs_paths, 6)
            # total_files = len(docs_paths)
            # map_obj = pool.map_async(partial_transform_docs, data)
            # consumed=0
            # while not map_obj.ready() or manager_queue.qsize() != 0:
            #     for _ in range(manager_queue.qsize()):
            #         manager_queue.get()
            #         consumed += 1
            #     sys.stderr.write('\rdone {0:%}'.format(consumed / total_files))
            # print()
            # result = map_obj.get()
            # for r in result:
            #     self.new_paths.update(r)

    def get_new_paths(self, files_list):
        if self.new_paths is not None and len(self.new_paths.keys()) != 0:
            new_list = []
            for file in files_list:
                if file in self.new_paths.keys():
                    new_list.append(self.new_paths[file])
            return new_list
        else:
            raise Exception("Data not transformed!")



class Word2VecEmbeddingCreator(object):

    """
    A class that transforms a text into their representation of word embedding
    It uses a trained word2vec model model to build a 3 dimentional vector representation of the document.
     The first dimension represents the document, the second dimension represents the word and the third dimension is the word embedding array
    """

    def __init__(self, word2vecModel, embeddingSize=200, window = 2):
        self.word2vecModel = word2vecModel
        self.embeddingSize = embeddingSize
        self.window = window

    def create_embedding_matrix(self, text, max_words=None):
        """
        Transform a tokenized text into a 2 dimensional array with the word2vec model
        :param text: the tokenized text
        :param max_words: the max number of words to put into the 3 dimensional array
        :return: the 3 dimensional array representing the content of the tokenized text
        """
        if max_words is None:
            x = np.zeros(shape=(len(text), self.embeddingSize), dtype='float')
        else:
            x = np.zeros(shape=(max_words, self.embeddingSize), dtype='float')
        for pos, w in enumerate(text):
            if max_words is not None and pos >= max_words:
                break
            try:
                x[pos] = self.word2vecModel.wv[w]
            except:
                # x[pos] = np.zeros(shape=self.embeddingSize)
                if pos - self.window < 0:
                    begin = 0
                else:
                    begin = pos - self.window
                if pos + self.window > len(text):
                    end = len(text)
                else:
                    end = pos + self.window
                word = self.word2vecModel.predict_output_word(text[begin:end])[0][0]
                x[pos] = self.word2vecModel.wv[word]
        return x

class EnsembleMetaLearnerDataCreator():

    def __init__(self, weak_classifiers, use_class_prediction=False):
        self.weak_classifiers = weak_classifiers
        self.representation_length = None
        self.new_paths = dict()
        if not use_class_prediction:
            print("Changing model structure")
            self.__change_weak_classifiers()


    def create_meta_learner_data(self, dataset, new_representation_path):
        """
        Transform representation from all dataset using the weak classifiers passed on constructor.
        Do it using multiprocessing
        :param dataset: the paths for the events as .csv files
        :param new_representation_path: the path where the new representations will be saved
        :return: None
        """
        if not os.path.exists(new_representation_path):
            os.mkdir(new_representation_path)
        self.new_paths = self.transform_representations(dataset, new_representation_path=new_representation_path)
        # with multiprocessing.Pool(processes=1) as pool:
        #     manager = multiprocessing.Manager()
        #     manager_queue = manager.Queue()
        #     partial_transform_representation = partial(self.transform_representations,
        #                                                new_representation_path=new_representation_path,
        #                                                 manager_queue=manager_queue)
        #     data = numpy.array_split(dataset, 6)
        #     # self.transform_representations(data[0], new_representation_path=new_representation_path, manager_queue=manager_queue)
        #     # exit()
        #     total_files = len(dataset)
        #     map_obj = pool.map_async(partial_transform_representation, data)
        #     consumed = 0
        #     while not map_obj.ready() or manager_queue.qsize() != 0:
        #         for _ in range(manager_queue.qsize()):
        #             manager_queue.get()
        #             consumed += 1
        #         sys.stderr.write('\rdone {0:%}'.format(consumed / total_files))
        #     result = map_obj.get()
        #     padded_paths = dict()
        #     for r in result:
        #         padded_paths.update(r)
        #     self.new_paths = padded_paths

    def transform_representations(self, dataset, new_representation_path=None, manager_queue=None):
        """
        Do the actual transformation
        :param dataset: the paths for the events as .csv files
        :param new_representation_path: the path where the new representations will be saved
        :param manager_queue: the multiprocessing.Manager.Queue to use for progress checking
        :return: dictionary {old_path : new_path}
        """
        new_paths = dict()
        consumed = 0
        total_files = len(dataset)
        for path in dataset:
            if isinstance(path, tuple):
                icustayid = os.path.splitext(path[0].split('/')[-1])[0]
            else:
                icustayid = os.path.splitext(path.split('/')[-1])[0]
            if manager_queue is not None:
                manager_queue.put(path)
            else:
                sys.stderr.write('\rdone {0:%}'.format(consumed / total_files))
            transformed_doc_path = new_representation_path + icustayid + '.pkl'
            if os.path.exists(transformed_doc_path):
                new_paths[icustayid] = transformed_doc_path
                if self.representation_length is None:
                    with open(transformed_doc_path, 'rb') as fhandler:
                        new_representation = pickle.load(fhandler)
                        self.representation_length = len(new_representation)
                continue
            data = self.__load_data(path)
            new_representation = self.__transform(data)
            if self.representation_length is None:
                self.representation_length = len(new_representation)
                print("Representation len {}".format(self.representation_length))
            with open(transformed_doc_path, 'wb') as fhandler:
                pickle.dump(new_representation, fhandler)
            new_paths[icustayid] = transformed_doc_path
            consumed += 1
        return new_paths

    def __load_data(self, path):
        if isinstance(path, tuple):
            data = []
            for p in path:
                if 'pkl' in p.split('.')[-1]:
                    with open(p, 'rb') as fhandler:
                        data.append(pickle.load(fhandler))
                elif 'csv' in p.split('.')[-1]:
                    data.append(pandas.read_csv(p).values)
            return data
        elif isinstance(path, str):
            if 'pkl' in path.split('.')[-1]:
                with open(path, 'rb') as fhandler:
                    return pickle.load(fhandler)
            elif 'csv' in path.split('.')[-1]:
                return pandas.read_csv(path).values

    def __transform(self, data):
        if isinstance(data, tuple):
            for data_index in range(len(data)):
                data[data_index] = np.array([data[data_index]])
        else:
            data = np.array([data])
        new_representation = []
        for model in self.weak_classifiers:
            if isinstance(model, tuple):
                data_index = model[1]
                model = model[0]
                prediction = model.predict(data[data_index])[0]
                new_representation.extend(prediction)
            else:
                prediction = model.predict(data)[0]
                new_representation.extend(prediction)
        return np.array(new_representation)

    def __change_weak_classifiers(self):
        new_weak_classifiers = []
        for model in self.weak_classifiers:
            if isinstance(model, tuple):
                new_model = Model(inputs=model[0].input, outputs=model[0].layers[-2].output)
                new_model.compile(loss=model[0].loss, optimizer=model[0].optimizer)
                new_model = (new_model, model[1])
            else:
                new_model = Model(inputs=model.input, outputs=model.layers[-2].output)
                new_model.compile(loss=model.loss, optimizer=model.optimizer)
            new_weak_classifiers.append(new_model)
        self.weak_classifiers = new_weak_classifiers

    def get_new_paths(self, files_list):
        if self.new_paths is not None and len(self.new_paths.keys()) != 0:
            new_list = []
            for file in files_list:
                if isinstance(file, tuple):
                    icustayid = os.path.splitext(file[0].split('/')[-1])[0]
                else:
                    icustayid = os.path.splitext(file.split('/')[-1])[0]
                if icustayid in self.new_paths.keys():
                    new_list.append(self.new_paths[icustayid])
            return new_list
        else:
            raise Exception("Data not transformed!")

class ClinicalBertTextRepresentationTransform():

    def __init__(self, transformed_text_saving_path:str):
        self.clinical_tokenizer = ClinicalTokenizer()
        self.bert_transformer = TransformTextsWithHuggingfaceBert()
        self.bert_transformer.load_clinical_bert()
        self.transformed_text_saving_path = transformed_text_saving_path

    def transform(self, data_df: pandas.DataFrame, text_paths_column:str, tokenization_strategy:str= "all", n_tokens:int=128,
                  sentence_encoding_strategy:str="mean", remove_temporal_axis:bool=False) -> pandas.DataFrame:
        """
        Do the transformation of clinical texts
        :param data_df: the dataset
        :param text_paths_column: the column with the paths where the text is stored
        :param tokenization_strategy: the text tokenization strategy: all sentences, only the first or last n_tokens in the texts
        :param n_tokens: the number of tokens used for the first and last tokenization strategy
        :param sentence_encoding_strategy: if use the mean over the sentences encoding to represent the text or to use only the [CLS] token
        :return: the new paths for the encoded representation files
        """
        if remove_temporal_axis:
            remove_no_text_constant = True
            document_mean = False
        else:
            remove_no_text_constant = False
            document_mean = True
        episodes = []
        new_paths = []
        labels = []
        consumed = 0
        total_files = len(data_df)
        for index, row in data_df.iterrows():
            sys.stderr.write('\rdone {0:%}'.format(consumed / total_files))
            consumed += 1
            episode_representation_path = self.__get_encoded_path(row['episode'])
            if os.path.exists(episode_representation_path):
                episodes.append(row['episode'])
                new_paths.append(episode_representation_path)
                labels.append(row['label'])
                continue
            texts_df = pandas.read_csv(row[text_paths_column], index_col='bucket').sort_index()
            ids_series:pandas.Series = self.clinical_tokenizer.process_texts_df_for_bert(texts_df,
                                                                                         text_strategy=tokenization_strategy,
                                                                                         n_tokens=n_tokens,
                                                                                         remove_no_text_constant=remove_no_text_constant)
            if ids_series is None:
                continue
            if remove_temporal_axis:
                ids_series = self.__remove_temporal_axis(ids_series)
            encoded_text_sequence = self.bert_transformer.transform_ids_series(ids_series,
                                                                               sentence_encoding_strategy=sentence_encoding_strategy,
                                                                               document_mean=document_mean)
            encoded_text_sequence = np.asarray(encoded_text_sequence.values.tolist())
            self.__save_encoded_text(episode_representation_path, encoded_text_sequence)
            new_paths.append(episode_representation_path)
            episodes.append(row['episode'])
            labels.append(row['label'])
        new_paths = pandas.DataFrame({'path':new_paths, 'label':labels}, index=episodes)
        return new_paths

    def __get_encoded_path(self, episode:str) -> str :
        return os.path.join(self.transformed_text_saving_path, '{}.pkl'.format(episode))

    def __save_encoded_text(self, episode_saving_path:str, encoded_representation):
        with open(episode_saving_path, 'wb') as file:
            pickle.dump(encoded_representation, file)

    def __remove_temporal_axis(self, ids_series:pandas.Series):
        new_series = []
        for index, document in ids_series.iteritems():
            for sentence in document:
                sentence = sentence.tolist()
                sentence = torch.as_tensor([sentence])
                new_series.append(sentence)
        new_series = pandas.Series(new_series)
        return new_series


class TransformTextsWithHuggingfaceBert():

    def __init__(self):
        self.model = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def load_clinical_bert(self):
        if self.model is None:
            self.model = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
            self.model = self.model.to(self.device)

    def transform_ids_series(self, ids_series:pandas.Series, sentence_encoding_strategy:str="mean", document_mean:bool=True):
        if self.model is None:
            self.load_clinical_bert()
        if sentence_encoding_strategy != "mean" and sentence_encoding_strategy != "cls":
            return None
        encoded_texts_sequence = pandas.Series([])
        for index, value in ids_series.iteritems():
            encoded_sentences = []
            for tensor in value:
                tensor = tensor.to(self.device)
                # print(tensor, tensor.shape)
                # print("------------")
                outputs = None
                try:
                    with torch.no_grad():
                        outputs = self.model(tensor)
                except Exception as e:
                    print(outputs)
                    print(self.model)
                    print(e)
                    exit()
                sentence_representation = None
                if sentence_encoding_strategy == "mean":
                    sentence_representation = self.__sentence_mean_strategy(outputs[0])
                elif sentence_encoding_strategy == "cls":
                    sentence_representation = self.__sentence_cls_strategy(outputs[0])
                if sentence_representation is None:
                    return None
                sentence_representation = sentence_representation.tolist()
                encoded_sentences.append(sentence_representation)
            if document_mean:
                encoded_sentences = np.mean(encoded_sentences, axis=0)
            else:
                new_encoded_sentences = []
                for sentence in encoded_sentences:
                    sentence = np.squeeze(sentence)
                    new_encoded_sentences.append(sentence)
                encoded_sentences = np.asarray(new_encoded_sentences)
            text_representation = pandas.Series([encoded_sentences], index=[index])
            encoded_texts_sequence = encoded_texts_sequence.append(text_representation)
        return encoded_texts_sequence

    def __sentence_mean_strategy(self, sentence_hidden_output:Tensor):
        mean = torch.mean(sentence_hidden_output, 1, keepdim=True)
        mean = torch.squeeze(mean)
        return mean

    def __sentence_cls_strategy(self, sentence_hidden_output:Tensor):
        return sentence_hidden_output[0][0]


class ClinicalTokenizer():

    def __init__(self):
        self.bert_tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
        self.sentence_tokenizer = spacy.load("en_core_sci_md")

    def process_texts_df_for_bert(self, texts_df:pandas.DataFrame, text_strategy:str="all", n_tokens:int=128,
                                  remove_no_text_constant=False):
        encoded_sentences = pandas.Series([])
        for index, row in texts_df.iterrows():
            encoded_texts = None
            if remove_no_text_constant:
                if row['text'] == constants.NO_TEXT_CONSTANT:
                    continue
            if text_strategy == "all":
                encoded_texts = self.__bert_encode_all_strategy(index, row['text'], n_tokens)
            elif text_strategy == "first":
                encoded_texts = self.__bert_encode_first_strategy(index, row['text'], n_tokens)
            elif text_strategy == "last":
                encoded_texts = self.__bert_encode_last_strategy(index, row['text'], n_tokens)
            encoded_sentences = encoded_sentences.append(encoded_texts)
        if encoded_sentences.empty:
            return None
        return encoded_sentences

    def __bert_encode_all_strategy(self, index, text:str, n_tokens:int):
        sentences = self.tokenize_sentences(text)
        sentences = self.bert_encode_sentences(sentences, n_tokens)
        return pandas.Series([sentences], index=[index])

    def __bert_encode_first_strategy(self, index, text:str, n_tokens:int):
        encoded_text = self.bert_encode_text(text)
        encoded_text = encoded_text.tolist()[0]
        if len(encoded_text) > n_tokens:
            encoded_text = [encoded_text[:n_tokens-1] + [encoded_text[-1]]]
        else:
            encoded_text = [encoded_text]
        encoded_text = torch.as_tensor([encoded_text])
        return pandas.Series([encoded_text], index=[index])

    def __bert_encode_last_strategy(self, index, text:str, n_tokens:int):
        encoded_text = self.bert_encode_text(text)
        encoded_text = encoded_text.tolist()[0]
        if len(encoded_text) > n_tokens:
            encoded_text = [[encoded_text[0]] + encoded_text[-(n_tokens - 1):]]
        else:
            encoded_text = [encoded_text]
        encoded_text = torch.as_tensor([encoded_text])
        return pandas.Series([encoded_text], index=[index])

    def tokenize_sentences(self, text:str):
        tokenized_sentences = self.sentence_tokenizer(text)
        return list(tokenized_sentences.sents)

    def bert_encode_sentences(self, sentences:[], n_tokens:int):
        encoded_sentences = []
        for sentence in sentences:
            encoded_sentences.append(self.bert_tokenizer.encode(str(sentence), return_tensors="pt", padding=True, truncation=True,
                                                                max_length=n_tokens))
        return encoded_sentences

    def bert_encode_text(self, text:str):
        encoded_text = self.bert_tokenizer.encode(text, return_tensors="pt")
        return encoded_text