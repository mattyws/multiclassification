import pickle
import uuid
import numpy as np
import os
from keras.utils import Sequence

from data_representation import Word2VecEmbeddingCreator


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