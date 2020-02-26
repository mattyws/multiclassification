import os
import pickle
import types
from functools import partial
from keras.models import Model

import multiprocessing

import numpy
import numpy as np
import pandas
import sys


class TransformClinicalTextsRepresentations(object):
    """
    Changes the representation for patients notes using a word2vec model.
    The patients notes must be into different csv.
    """
    def __init__(self, word2vec_model, embedding_size=200, text_max_len=None,  window=2, texts_path=None, representation_save_path=None):
        self.word2vec_model = word2vec_model
        self.embedding_size = embedding_size
        self.window = window
        self.texts_path = texts_path
        self.text_max_len = text_max_len
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
                x.append(self.word2vec_model.wv[w])
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
                    word = self.word2vec_model.predict_output_word(text[begin:end])[0][0]
                    # x[pos] = self.word2vec_model.wv[word]
                    x.append(self.word2vec_model.wv[word])
                except:
                    # continue
                    # x[pos] = np.zeros(shape=self.embedding_size)
                    x.append(np.zeros(shape=self.embedding_size))
        x = np.array(x)
        return x

    def transform_docs(self, docs_path, preprocessing_pipeline=[], manager_queue=None):
        new_paths = dict()
        for path in docs_path:
            file_name = path.split('/')[-1]
            if manager_queue is not None:
                manager_queue.put(path)
            transformed_doc_path = self.representation_save_path + os.path.splitext(file_name)[0] + '.pkl'
            if os.path.exists(transformed_doc_path):
                new_paths[path] = transformed_doc_path
                continue
            data = pandas.read_csv(path)
            transformed_texts = []
            for index, row in data.iterrows():
                try:
                    note = row['Note']
                except Exception as e:
                    print(path)
                    raise Exception("deu errado")
                if preprocessing_pipeline is not None:
                    for func in preprocessing_pipeline:
                        note = func(note)
                new_representation = self.create_embedding_matrix(note)
                if new_representation is not None:
                    transformed_texts.append(new_representation)
            if len(transformed_texts) != 0:
                transformed_texts = numpy.array(transformed_texts)
                with open(transformed_doc_path, 'wb') as handler:
                    pickle.dump(transformed_texts, handler)
                new_paths[path] = transformed_doc_path
        return new_paths

    def transform(self, docs_paths, preprocessing_pipeline=None):
        with multiprocessing.Pool(processes=4) as pool:
            manager = multiprocessing.Manager()
            manager_queue = manager.Queue()
            self.lock = manager.Lock()
            partial_transform_docs = partial(self.transform_docs,
                                             preprocessing_pipeline=preprocessing_pipeline,
                                             manager_queue=manager_queue)
            data = numpy.array_split(docs_paths, 6)
            total_files = len(docs_paths)
            map_obj = pool.map_async(partial_transform_docs, data)
            consumed=0
            while not map_obj.ready() or manager_queue.qsize() != 0:
                for _ in range(manager_queue.qsize()):
                    manager_queue.get()
                    consumed += 1
                sys.stderr.write('\rdone {0:%}'.format(consumed / total_files))
            print()
            result = map_obj.get()
            for r in result:
                self.new_paths.update(r)

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
        data = np.array([data])
        new_representation = []
        for model in self.weak_classifiers:
            if isinstance(model, tuple):
                data_index = model[1]
                model = model[0]
                prediction = model.predict(data[data_index])
                new_representation.extend(prediction)
            else:
                prediction = model.predict(data)[0]
                new_representation.extend(prediction)
        return new_representation

    def __change_weak_classifiers(self):
        new_weak_classifiers = []
        for model in self.weak_classifiers:
            if isinstance(model, tuple):
                new_model = Model(inputs=model[0].input, outputs=model[0].layers[-2].output)
                new_model = (new_model, model[1])
            else:
                new_model = Model(inputs=model.input, outputs=model.layers[-2].output)
            new_model.compile(loss=model.loss, optimizer=model.optimizer)
            new_weak_classifiers.append(new_model)
        self.weak_classifiers = new_weak_classifiers

    def get_new_paths(self, files_list):
        # TODO: considerar os tipos de dados
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


class AutoencoderDataCreator():

    def __init__(self, encoder):
        self.new_paths = []
        self.encoder = encoder
        pass

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
            with open(transformed_doc_path, 'wb') as fhandler:
                pickle.dump(new_representation, fhandler)
            new_paths[icustayid] = transformed_doc_path
            consumed += 1
        return new_paths

    def create_autoencoder_representation(self, dataset, new_representation_path=None):
        with multiprocessing.Pool(processes=1) as pool:
            manager = multiprocessing.Manager()
            manager_queue = manager.Queue()
            partial_transform_representation = partial(self.transform_representations,
                                                       new_representation_path=new_representation_path,
                                                        manager_queue=manager_queue)
            data = numpy.array_split(dataset, 6)
            # self.transform_representations(data[0], new_representation_path=new_representation_path, manager_queue=manager_queue)
            # exit()
            total_files = len(dataset)
            map_obj = pool.map_async(partial_transform_representation, data)
            consumed = 0
            while not map_obj.ready() or manager_queue.qsize() != 0:
                for _ in range(manager_queue.qsize()):
                    manager_queue.get()
                    consumed += 1
                sys.stderr.write('\rdone {0:%}'.format(consumed / total_files))
            result = map_obj.get()
            padded_paths = dict()
            for r in result:
                padded_paths.update(r)
            self.new_paths = padded_paths

    def __load_data(self, path):
        if 'pkl' in path.split('.')[-1]:
            with open(path, 'rb') as fhandler:
                return pickle.load(fhandler)
        elif 'csv' in path.split('.')[-1]:
            return pandas.read_csv(path).values

    def __transform(self, data):
        data = np.array([data])
        prediction = self.encoder.predict(data)
        return prediction

    def get_new_paths(self, files_list):
        if self.new_paths is not None and len(self.new_paths.keys()) != 0:
            new_list = []
            for file in files_list:
                if file in self.new_paths.keys():
                    new_list.append(self.new_paths[file])
            return new_list
        else:
            raise Exception("Data not transformed!")