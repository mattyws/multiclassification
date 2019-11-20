import os
import pickle
from functools import partial

import multiprocessing

import numpy
import numpy as np
import pandas
import sys


def create_embedding_matrix(text, word2vec_model, embedding_size, window):
    """
    Transform a tokenized text into a 3 dimensional array with the word2vec model
    :param text: the tokenized text
    :return: the 3 dimensional array representing the content of the tokenized text
    """
    # x = np.zeros(shape=(len(text), embedding_size), dtype='float')
    x = []
    for pos, w in enumerate(text):
        try:
            # x[pos] = word2vec_model.wv[w]
            x.append(word2vec_model.wv[w])
        except:
            # x[pos] = np.zeros(shape=self.embeddingSize)
            if pos - window < 0:
                begin = 0
            else:
                begin = pos - window
            if pos + window > len(text):
                end = len(text)
            else:
                end = pos + window
            try:
                word = word2vec_model.predict_output_word(text[begin:end])[0][0]
                # x[pos] = word2vec_model.wv[word]
                x.append(word2vec_model.wv[word])
            except:
                # x[pos] = np.zeros(shape=embedding_size)
                x.append(np.zeros(shape=embedding_size))
    x = np.array(x)
    return x

def transform_docs(docs_path, word2vec_model, embedding_size, window, representation_save_path, preprocessing_pipeline=[],
                   manager_queue=None):
    new_paths = dict()
    for path in docs_path:
        file_name = path.split('/')[-1]
        if manager_queue is not None:
            manager_queue.put(path)
        transformed_doc_path = representation_save_path + os.path.splitext(file_name)[0] + '.pkl'
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
            new_representation = create_embedding_matrix(note, word2vec_model, embedding_size, window)
            transformed_texts.append(new_representation)
        transformed_texts = numpy.array(transformed_texts)
        with open(transformed_doc_path, 'wb') as handler:
            pickle.dump(transformed_texts, handler)
        new_paths[path] = transformed_doc_path
    return new_paths


class TransformClinicalTextsRepresentations(object):
    """
    Changes the representation for patients notes using a word2vec model.
    The patients notes must be into different csv.
    """
    def __init__(self, word2vec_model, embedding_size=200, window=2, texts_path=None, representation_save_path=None):
        self.word2vec_model = word2vec_model
        self.embedding_size = embedding_size
        self.window = window
        self.texts_path = texts_path
        self.representation_save_path = representation_save_path
        if not os.path.exists(representation_save_path):
            os.mkdir(representation_save_path)
        self.new_paths = dict()

    def transform(self, docs_paths, preprocessing_pipeline=None):
        with multiprocessing.Pool(processes=4) as pool:
            manager = multiprocessing.Manager()
            manager_queue = manager.Queue()
            partial_transform_docs = partial(transform_docs, word2vec_model=self.word2vec_model,
                                             embedding_size=self.embedding_size, window=self.window,
                                             representation_save_path=self.representation_save_path,
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

    def get_new_paths(self, files_list):
        if self.new_paths is not None:
            new_list = []
            for file in files_list:
                new_list.append(self.new_paths[file])
            return new_list
        else:
            raise Exception("Data not normalized!")


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
