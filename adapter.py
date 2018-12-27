import abc
import pickle

import keras
import numpy as np
import itertools

from keras.models import load_model


class ModelAdapter(object, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def fit(self, trainDocs, trainCats, epochs=0, batch_size=10):
        raise NotImplementedError('users must define \'fit\' to use this base class')

    @abc.abstractmethod
    def predict(self, testDocs, batch_size=10):
        raise NotImplementedError('users must define \'predict\' to use this base class')

    @abc.abstractmethod
    def predict_one(self, doc):
        raise NotImplementedError('users must define \'predict_one\' to use this base class')

    @abc.abstractmethod
    def save(self, filename):
        raise NotImplementedError('users must define \'save\' to use this base class')

    @abc.abstractmethod
    def load(self, filename):
        raise NotImplementedError('users must define \'load\' to use this base class')

class KerasGeneratorAdapter(ModelAdapter):

    def __init__(self, model):
        self.model = model

    def fit(self, trainDocs, trainCats, epochs=1, batch_size=10, workers=2, validationDocs=None, validationLabels=None,
            validationSteps=None):
        data = self.XYGenerator(trainDocs, trainCats)
        validationData = None
        if validationDocs is not None and validationLabels is not None:
            validationData = self.XYGenerator(validationDocs, validationLabels)
        self.model.fit_generator(data, batch_size, epochs=epochs, initial_epoch=0, max_q_size=1, verbose=1,
                                 workers=workers, validation_data=validationData, validation_steps=validationSteps)

    def predict(self, testDocs, batch_size=10):
        result = []
        for data in testDocs:
            result.append(self.model.predict_classes(data, batch_size, verbose=1))
        return result

    def predict_one(self, doc):
        result = self.model.predict(doc)
        return np.argmax(result)

    def predict_one_array(self,doc):
        result = self.model.predict(doc)
        return result

    def save(self, filename):
        self.model.save(filename)

    def load(self, filename):
        return KerasGeneratorAdapter(load_model(filename))

    class XYGenerator(object):
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

class SklearnAdapter(ModelAdapter):

    def __init__(self, model):
        self.model = model

    def predict(self, testDocs, batch_size=10):
        pred = self.model.predict(testDocs)
        return pred

    def fit(self, trainDocs, trainCats, epochs=0, batch_size=10):
        self.model.fit(trainDocs, trainCats)

    def fit_generator(self, data_generator, epochs=0, batch_size=10):
        raise NotImplementedError('\'fit_generator\' not implemented in this class.')

    def predict_one(self, doc):
        return self.model.predict(doc)

    def save(self, filename):
        pickle.dump(self.model, open(filename, 'wb'))

    def load(self, filename):
        return pickle.load(open(filename, 'rb'))
