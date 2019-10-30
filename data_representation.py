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
    x = np.zeros(shape=(len(text), embedding_size), dtype='float')
    for pos, w in enumerate(text):
        try:
            x[pos] = word2vec_model.wv[w]
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
            word = word2vec_model.predict_output_word(text[begin:end])[0][0]
            x[pos] = word2vec_model.wv[word]
    return x

def transform_docs(icustays, word2vec_model, embedding_size, window, texts_path, representation_save_path, manager_queue=None):
    for icustay in icustays:
        if manager_queue is not None:
            manager_queue.add(icustay)
        transformed_doc_path = representation_save_path + icustay + '/'
        if not os.path.exists(texts_path + icustay + '.csv'):
            continue
        if not os.path.exists(transformed_doc_path):
            os.mkdir(transformed_doc_path)
        data = pandas.read_csv(texts_path + icustay + '.csv')
        for index, row in data.iterrows():
            new_representation = create_embedding_matrix(row['Note'], word2vec_model, embedding_size, window)
            with open(transformed_doc_path + "{}_{}.pkl".format(index, row['timestamp']), 'wb') as handler:
                pickle.dump(new_representation, handler)


class TransformClinicalTextsRepresentations(object):
    """
    Changes the representation for patients notes using a word2vec model.
    The patients notes must be into different csv.
    """
    def __init__(self, word2vecModel, embeddingSize=200, window=2, texts_path=None, representation_save_path=None):
        self.word2vecModel = word2vecModel
        self.embeddingSize = embeddingSize
        self.window = window
        self.texts_path = texts_path
        self.representation_save_path = representation_save_path

    def transform(self, icustays):
        with multiprocessing.Pool(processes=4) as pool:
            manager = multiprocessing.Manager()
            manager_queue = manager.Queue()
            partial_transform_docs = partial(transform_docs, word2vecModel=self.word2vecModel,
                                             embeddingSize=self.embeddingSize, window=self.window,
                                             texts_path=self.texts_path, representation_save_path=self.representation_save_path,
                                             manager_queue=manager_queue)
            data = numpy.array_split(icustays, 6)
            total_files = len(icustays)
            map_obj = pool.map_async(partial_transform_docs, data)
            consumed=0
            while not map_obj.ready() or manager_queue.qsize() != 0:
                for _ in range(manager_queue.qsize()):
                    manager_queue.get()
                    consumed += 1
                sys.stderr.write('\rdone {0:%}'.format(consumed / total_files))


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
