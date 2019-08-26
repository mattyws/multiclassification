import json
import pickle
import uuid
from math import ceil

import numpy as np
import os
import pandas as pd
from keras.utils import Sequence

from data_representation import Word2VecEmbeddingCreator

class EmbeddingObjectSaver(object):
    def __init__(self, path):
        self.path = path
        if self.path[-1] != '/':
            self.path += '/'
        if not os.path.exists(self.path):
            os.mkdir(self.path)

    def save(self, embeddingObject):
        return self.__save_object(embeddingObject)

    def __save_object(self, embeddingObject):
        fileName = self.path + uuid.uuid4().hex
        while os.path.exists(fileName):
            fileName = self.path +  uuid.uuid4().hex
        fileName = fileName + '.pkl'
        with open(fileName, 'wb+') as pickleFileHandler:
            try:
                pickle.dump(embeddingObject, pickleFileHandler, pickle.HIGHEST_PROTOCOL)
            except Exception as e:
                print(e)
        return fileName

class EmbeddingObjectLoader(object):

    def load(self, filePath):
        return self.__load(filePath)

    def __load(self, fileName):
        x = None
        with open(fileName, 'rb') as pklFileHandler:
            x = pickle.load(pklFileHandler)
        return x

class EmbeddingObjectsDelete(object):

    def __init__(self, paths):
        self.__filesList = paths

    def clean_files(self):
        for file in self.__filesList:
            os.remove(file)

class LengthLongitudinalDataGenerator(Sequence):
    def __init__(self, sizes_data_paths, labels, max_batch_size=50, iterForever=False):
        self.max_batch_size = max_batch_size
        self.batches = sizes_data_paths
        self.labels = labels
        self.iterForever = iterForever
        self.__iterPos = 0

    def create_batches(self):
        new_batches = dict()
        new_labels = dict()
        batch_num = 0
        for key in self.batches.keys():
            split_data = np.array_split(self.batches[key], ceil(len(self.batches[key])/self.max_batch_size) )
            split_classes = np.array_split(self.labels[key], ceil(len(self.labels[key]) / self.max_batch_size))
            for s, c in zip(split_data, split_classes):
                new_batches[batch_num] = s
                new_labels[batch_num] = c
                batch_num += 1
        self.batches = new_batches
        self.labels = new_labels

    def __load(self, filesNames):
        x = []
        max_len = None
        columns_len = None
        for fileName in filesNames:
            with open(fileName, 'rb') as data_file:
                data = pickle.load(data_file)
            x.append(data)
            # if max_len is None or len(data) > max_len:
            #     max_len = len(data)
            # if columns_len is None:
            #     columns_len = len(data[0])
        # # Zero padding the matrices
        # zero_padding_x = []
        # i = 0
        # for value in x:
        #     i += 1
        #     zeros = np.zeros((max_len, columns_len))
        #     zeros[:value.shape[0], : value.shape[1]] = value
        #     zero_padding_x.append(zeros)
        x = np.array(x)
        return x

    def __iter__(self):
        return self

    def __getitem__(self, idx):
        """
        :param idx:
        :return:
        """
        # if self.__batch_exists(idx):
        #     batch_x, batch_y = self.__load_batch(idx)
        # else:
        batch_x = self.batches[idx]
        batch_x = self.__load(batch_x)
        batch_y = self.labels[idx]
        # self.__save_batch(idx, batch_x, batch_y)
        return batch_x, batch_y

    def __len__(self):
        return len(self.batches.keys())


class LongitudinalDataGenerator(Sequence):

    def __init__(self, dataPaths, labels, batchSize, iterForever=False, saved_batch_dir='saved_batch/'):
        self.batchSize = batchSize
        self.__labels = labels
        self.__filesList = dataPaths
        self.iterForever = iterForever
        self.__iterPos = 0
        if not saved_batch_dir.endswith('/'):
            saved_batch_dir += '/'
        if not os.path.exists(saved_batch_dir):
            os.mkdir(saved_batch_dir)
        self.saved_batch_dir = saved_batch_dir

    def __load(self, filesNames):
        x = []
        max_len = None
        columns_len = None
        for fileName in filesNames:
            with open(fileName, 'rb') as data_file:
                data = pickle.load(data_file)
            x.append(data)
            if max_len is None or len(data) > max_len:
                max_len = len(data)
            if columns_len is None:
                columns_len = len(data[0])
        # Zero padding the matrices
        zero_padding_x = []
        i = 0
        for value in x:
            i += 1
            zeros = np.zeros((max_len, columns_len))
            zeros[:value.shape[0], : value.shape[1]] = value
            zero_padding_x.append(zeros)
        x = np.array(zero_padding_x)
        return x

    def __save_batch(self, idx, batch_x, batch_y):
        with open(self.saved_batch_dir+'batch_{}.pkl'.format(idx), 'wb') as batch_file:
            pickle.dump((batch_x, batch_y), batch_file, protocol=4)

    def __load_batch(self, idx):
        with open(self.saved_batch_dir+'batch_{}.pkl'.format(idx), 'rb') as batch_file:
            data = pickle.load(batch_file)
            return data

    def __batch_exists(self, idx):
        return os.path.exists(self.saved_batch_dir+'batch_{}.pkl'.format(idx))

    def __iter__(self):
        return self

    def __getitem__(self, idx):
        """
        :param idx:
        :return:
        """
        # if self.__batch_exists(idx):
        #     batch_x, batch_y = self.__load_batch(idx)
        # else:
        batch_x = self.__filesList[idx * self.batchSize:(idx + 1) * self.batchSize]
        batch_x = self.__load(batch_x)
        batch_y = self.__labels[idx * self.batchSize:(idx + 1) * self.batchSize]
        # self.__save_batch(idx, batch_x, batch_y)
        return batch_x, batch_y

    def __len__(self):
        return np.int64(np.ceil(len(self.__filesList) / float(self.batchSize)))

class AutoencoderDataGenerator(Sequence):
    def __init__(self, dataPaths, iterForever=False):
        self.__filesList = dataPaths
        self.iterForever = iterForever
        self.__iterPos = 0
        self.total_events = 0
        self.consumed_idx = []

    def __load(self, filesNames):
        x = []
        for fileName in filesNames:
            with open(fileName, 'rb') as data_file:
                x = pickle.load(data_file)
        return x

    def __iter__(self):
        return self

    def __getitem__(self, idx):
        """
        :param idx:
        :return:
        """
        batch_x = self.__filesList[idx]
        batch_x = self.__load(batch_x)
        if idx not in self.consumed_idx:
            self.consumed_idx.append(idx)
            self.total_events += len(batch_x)
        return batch_x, batch_x

    def __len__(self):
        return len(self.__filesList)

class Word2VecTextEmbeddingGenerator(Sequence):
    def __init__(self, dataPath, word2vecModel, batchSize, embeddingSize=200, iterForever=False):
        self.dataPath = dataPath
        if self.dataPath[-1] != '/':
            self.dataPath += '/'
        if not os.path.exists(self.dataPath):
            os.mkdir(self.dataPath)
        self.word2vecGenerator = Word2VecEmbeddingCreator(word2vecModel, embeddingSize)
        self.iterForever = iterForever
        self.batchSize = batchSize
        self.__filesList = []
        self.__labels = []
        self.__iterPos = 0

    def add(self, text, label, maxWords=None):
        embeddingObject = self.word2vecGenerator.create_embedding_matrix(text, max_words=maxWords)
        fileName = self.__save_object(embeddingObject)
        self.__filesList.append(fileName)
        self.__labels.append(label)
        # print(len(self.__labels), len(self.__filesList))

    def clean_files(self):
        for file in self.__filesList:
            os.remove(file)
        self.__filesList = []
        self.__labels = []


    def __save_object(self, embeddingObject):
        fileName = uuid.uuid4().hex
        while fileName in self.__filesList:
            fileName = uuid.uuid4().hex
        fileName = self.dataPath+fileName+'.pkl'
        with open(fileName, 'wb+') as pickleFileHandler:
            try:
                pickle.dump(embeddingObject, pickleFileHandler, pickle.HIGHEST_PROTOCOL)
            except Exception as e:
                print(e)
        return fileName

    def __load(self, fileName):
        x = None
        with open(fileName, 'rb') as pklFileHandler:
            x = pickle.load(pklFileHandler)
        return np.array(x)


    def __iter__(self):
        return self

    def __getitem__(self, idx):
        """
        Its assumed that we deliver only one data at each batch
        :param idx:
        :return:
        """
        # batch_x = self.__filesList[idx * self.batchSize:(idx + 1) * self.batchSize]
        # batch_y = self.__labels[idx * self.batchSize:(idx + 1) * self.batchSize]
        batch_x = np.array([self.__load(self.__filesList[idx])])
        batch_y = np.array(self.__labels[idx])

        # return np.array([self.__load(file_name) for file_name in batch_x]), np.array(batch_y)
        return batch_x, batch_y

    # def __next__(self):
    #     if len(self.__filesList) == 0:
    #         return None
    #     if self.__iterPos >= len(self.__filesList):
    #         if self.iterForever:
    #             self.__iterPos = 0
    #         else:
    #             raise StopIteration()
    #     print('\n', self.__iterPos, len(self.__labels))
    #     x = np.array(self.__load(self.__filesList[self.__iterPos]))
    #     y = self.__labels[self.__iterPos]
    #     self.__iterPos += 1
    #     return x, y


    def __len__(self):
        return np.int64(np.ceil(len(self.__filesList) / float(self.batchSize)))


class ListGenerator(object):
    def __init__(self, train_docs, train_cats):
        self.__docs = train_docs
        self.__labels = train_cats
        self.train_docs = iter(train_docs)
        self.train_cats = iter(train_cats)

    def __iter__(self):
        return self

    def __next__(self):
        try:
            x = np.array(next(self.train_docs))
            y = np.array(next(self.train_cats))
            return x, y
        except:
            self.train_docs = iter(self.__docs)
            self.train_cats = iter(self.__labels)
            return self.__next__()

    def next(self):
        return self.__next__()