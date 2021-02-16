import json
import pickle
import uuid
from math import ceil

import bert
import numpy as np
import os

import pandas
import pandas as pd
import tensorflow
from gensim.models.doc2vec import TaggedDocument
from keras.preprocessing.sequence import pad_sequences
from keras.utils import Sequence
from ast import literal_eval

from nltk import WhitespaceTokenizer

from resources.data_representation import Word2VecEmbeddingCreator, ClinicalTokenizer

from tensorflow.python.keras.utils.data_utils import Sequence as tsSeq



class ArrayDataGenerator(tsSeq):

    def __init__(self, data, labels, batch_size):
        self.data = data
        self.labels = labels
        self.batch_size = batch_size
        self.batches = []

    def __iter__(self):
        return self

    def __getitem__(self, idx):
        """
        :param idx:
        :return:
        """
        batch = []
        for i in range(idx * self.batch_size, (idx + 1) * self.batch_size):
            batch.append(i)
        self.batches.append(batch)
        batch_x = self.data[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_x = batch_x.astype('float64')
        batch_y = self.labels[idx * self.batch_size:(idx + 1) * self.batch_size]
        # print(batch_x, batch_y)
        return batch_x, batch_y

    def __len__(self):
        return np.int64(np.ceil(len(self.data) / float(self.batch_size)))

class BertDataGenerator(tsSeq):

    def __init__(self, data_paths, labels, batch_size):
        self.data_paths = data_paths
        self.labels = labels
        self.batch_size = batch_size

    def __load_files(self, filesNames):
        # print("load files")
        x = []
        for fileName in filesNames:
            with open(fileName, 'rb') as data_file:
                x.append(pickle.load(data_file))
        x = np.asarray(x)
        # print(x)
        return x

    def __iter__(self):
        return self

    def __getitem__(self, idx):
        """
        :param idx:
        :return:
        """
        batch_x = self.data_paths[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_x = self.__load_files(batch_x)
        batch_y = self.labels[idx * self.batch_size:(idx + 1) * self.batch_size]
        # print(batch_x, batch_y)
        return batch_x, batch_y

    def __len__(self):
        return np.int64(np.ceil(len(self.data_paths) / float(self.batch_size)))

class LengthLongitudinalDataGenerator(tsSeq):

    def __init__(self, sizes_data_paths, labels, max_batch_size=50, iterForever=False, ndmin=None):
        self.max_batch_size = max_batch_size
        self.batches = sizes_data_paths
        self.labels = labels
        self.iterForever = iterForever
        self.__iterPos = 0
        self.ndmin = ndmin

    def create_batches(self):
        new_batches = dict()
        new_labels = dict()
        batch_num = 0
        for key in self.batches.keys():
            split_data = np.array_split(self.batches[key], ceil(len(self.batches[key])/self.max_batch_size))
            split_classes = np.array_split(self.labels[key], ceil(len(self.labels[key]) / self.max_batch_size))
            for s, c in zip(split_data, split_classes):
                new_batches[batch_num] = s
                new_labels[batch_num] = c
                batch_num += 1
        self.batches = new_batches
        self.labels = new_labels
        # print(self.batches)

    def __load(self, filesNames):
        x = []
        for fileName in filesNames:
            with open(fileName, 'rb') as data_file:
                data = pickle.load(data_file)
            data = np.asarray(data).astype('float32')
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
        try:
            if self.ndmin is not None:
                x = np.array(x, ndmin=self.ndmin)
            else:
                x = np.array(x)
        except Exception as e:
            print(x)
            print(filesNames)
            print(e)
            x = []
            for fileName in filesNames:
                print(fileName)
                with open(fileName, 'rb') as data_file:
                    data = pickle.load(data_file)
                x.append(data)
                try:
                    np.array(x)
                except:
                    print(data)
            exit()
        # print("data generator: {}".format(filesNames))
        # print("data generator: {}".format(x.shape))
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
        # print(batch_x)
        # print("{} : =====".format(idx))
        # print(self.batches[idx])
        # for x in batch_x:
        #     print(x.shape)
        # print("=====")
        # self.__save_batch(idx, batch_x, batch_y)
        return batch_x, batch_y

    def __len__(self):
        return len(self.batches.keys())


class MixedLengthDataGenerator(tsSeq):

    def __init__(self, data_df:pd.DataFrame, max_batch_size:int=50, structured_df_column:str="",
                 textual_df_column:str="", structured_model_input_name:str="",
                 textual_model_input_name:str="", ndmin=None):
        self.max_batch_size = max_batch_size
        self.data_df = data_df
        self.ndmin = ndmin
        self.batches = None
        self.structured_df_column = structured_df_column
        self.textual_df_column = textual_df_column
        self.structured_model_input_name = structured_model_input_name
        self.textual_model_input_name = textual_model_input_name

    def create_batches(self, sizes):
        new_batches = dict()
        batch_num = 0
        for size in sizes.keys():
            split_episodes = np.array_split(sizes[size], ceil(len(sizes[size])/self.max_batch_size))
            for piece in split_episodes:
                batch_df = self.data_df[self.data_df['episode'].isin(piece)]
                new_batches[batch_num] = batch_df
                batch_num += 1
        self.batches = new_batches

    def __load(self, data_df:pd.DataFrame):
        structured_data = []
        textual_data = []
        instances = []
        for index, row in data_df.iterrows():
            instance = []
            with open(row[self.structured_df_column], 'rb') as data_file:
                # print("===== Structured =====")
                data = np.asarray(pickle.load(data_file))
                    # new_data.append(np.asarray(d))
                # data = np.asarray(new_data)
                instance.append(data)
                structured_data.append(data)
                # instance.append(data)
            with open(row[self.textual_df_column], 'rb') as data_file:
                # print("===== Textual =====")
                data = np.asarray(pickle.load(data_file))
                instance.append(data)
                textual_data.append(data)
                # instance.append(data)
            # instance = np.asarray(instance)
            instances.append(instance)
            # x.append(instance)
        try:
            structured_data = np.asarray(structured_data)
            # for d in structured_data:
            #     print(d.shape)
            # print(structured_data)
            # print("========================================================================")
            textual_data = np.asarray(textual_data)
            # for d in textual_data:
            #     print(d.shape)
            # print(textual_data)
        except Exception as e:
            print(e)
            print(data_df)
            exit()
        # return [structured_data, textual_data]
        # instances = np.asarray(instances)
        # instances = np.hstack((structured_data, textual_data))
        return {self.structured_model_input_name: structured_data, self.textual_model_input_name:textual_data}

    def __iter__(self):
        return self

    def __getitem__(self, idx):
        """
        :param idx:
        :return:
        """
        batch_df = self.batches[idx]
        # print(batch_df)
        batch_x = self.__load(batch_df)
        batch_y = np.asarray(batch_df['label'].tolist())
        return batch_x, batch_y

    def __len__(self):
        return len(self.batches.keys())


class NoteeventsLengthLongitudinalDataGenerator(LengthLongitudinalDataGenerator):
    """
    Same as LengthLongitudinalDataGenerator, but uses padding on texts
    """
    def __init__(self, sizes_data_paths, labels, max_batch_size=50, iterForever=False, pad_sequences=False,
                 max_pad_len=None):
        self.max_batch_size = max_batch_size
        self.batches = sizes_data_paths
        self.labels = labels
        self.iterForever = iterForever
        self.pad_sequences=pad_sequences
        self.max_pad_len=max_pad_len
        self.__iterPos = 0

    def __load(self, filesNames):
        x = []
        max_len = None
        columns_len = None
        for fileName in filesNames:
            with open(fileName, 'rb') as data_file:
                data = pickle.load(data_file)
            if self.pad_sequences:
                new_data = []
                for value in data:
                    value = pad_sequences(value, maxlen=self.max_pad_len)
                    new_data.append(value)
                data = new_data
            x.append(data)
        x = np.array(x)
        return x


class LongitudinalDataGenerator(tsSeq):

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

class Word2VecTextEmbeddingGenerator(tsSeq):
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


class NoteeventsTextDataGenerator(object):

    def __init__(self, data_paths, preprocessing_pipeline=None):
        self.data_paths = data_paths
        self.preprocessing_pipeline = preprocessing_pipeline

    def __iter__(self):
        tokenizer = WhitespaceTokenizer()
        for index, path in enumerate(self.data_paths):
            with open(path, 'r') as handler:
                for line in handler:
                    yield tokenizer.tokenize(line)
        return self

class TaggedNoteeventsDataGenerator(object):

    def __init__(self, data_paths, preprocessing_pipeline=None):
        self.data_paths = data_paths
        self.preprocessing_pipeline = preprocessing_pipeline
        self.clinical_tokenizer = ClinicalTokenizer()

    def __iter__(self):
        for index, path in enumerate(self.data_paths):
            patient_noteevents = pandas.read_csv(path)
            patient_id = os.path.basename(path).split('.')[0]
            for index, row in patient_noteevents.iterrows():
                text = row['text']
                sentences = self.clinical_tokenizer.tokenize_sentences(text)
                for num, sentence in enumerate(sentences):
                    processed_sentence = sentence
                    if self.preprocessing_pipeline is not None:
                        for func in self.preprocessing_pipeline:
                            processed_sentence = func(processed_sentence)
                    tagged_doc = TaggedDocument(words=processed_sentence, tags=["{}_{}_{}".format(patient_id,row['starttime'], num)])
                    yield tagged_doc
        return self