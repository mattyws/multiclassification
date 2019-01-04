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

    @abc.abstractmethod
    def evaluate(self, testDocs, batch_size=10):
        raise NotImplementedError('users must define \'evaluate\' to use this base class')

class KerasGeneratorAdapter(ModelAdapter):

    def __init__(self, model):
        self.model = model

    def fit(self, dataGenerator, epochs=1, batch_size=10, workers=2, validationDataGenerator = None,
            validationSteps=None, callbacks=None):
        self.model.fit_generator(dataGenerator, batch_size, epochs=epochs, initial_epoch=0, max_queue_size=1, verbose=1,
                                 workers=workers, validation_data=validationDataGenerator, validation_steps=validationSteps,
                                 callbacks=callbacks)

    def predict(self, testDocs, batch_size=10):
        result = []
        for data in testDocs:
            result.append(self.model.predict_classes(data, batch_size, verbose=1))
        return result

    def predict_one(self, doc):
        result = self.model.predict(doc)
        return np.argmax(result)

    def evaluate(self, testDocs, batch_size=10):
        result = self.model.evaluate_generator(testDocs, batch_size)
        result = {self.model.metrics_names[i] : result[i] for i in range(len(result)) }
        return result

    def predict_one_array(self,doc):
        result = self.model.predict(doc)
        return result

    def save(self, filename):
        self.model.save(filename)

    def load(self, filename):
        return KerasGeneratorAdapter(load_model(filename))

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
