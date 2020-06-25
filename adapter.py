import abc
import pickle
import numpy as np

import sys

from gensim.models import Word2Vec, Doc2Vec
from tensorflow.keras.models import load_model

from result_evaluation import ModelEvaluation


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
    def predict_generator(self, generator):
        raise NotImplementedError('users must define \'predict_one\' to use this base class')

    @abc.abstractmethod
    def save(self, filename):
        raise NotImplementedError('users must define \'save\' to use this base class')

    @abc.abstractmethod
    def evaluate(self, testDocs, batch_size=10):
        raise NotImplementedError('users must define \'evaluate\' to use this base class')

class KerasAdapter(ModelAdapter):

    def __init__(self, model):
        self.model = model

    def fit(self, dataGenerator, epochs=1, batch_size=10, workers=4, validationDataGenerator = None,
            validationSteps=None, callbacks=None, use_multiprocessing=True, class_weights=None):
        self.model.fit_generator(dataGenerator, len(dataGenerator), epochs=epochs, initial_epoch=0, max_queue_size=1, verbose=1,
                                 workers=workers, validation_data=validationDataGenerator, validation_steps=validationSteps,
                                 callbacks=callbacks, use_multiprocessing=use_multiprocessing, class_weight=class_weights)

    def predict(self, testDocs, batch_size=10):
        # result = self.model.predict(testDocs, batch_size, verbose=0)
        # result = result.argmax(axis=-1)
        # result = self.model.predict_generator(testDocs)
        result = self.model.predict(testDocs)
        # result = np.argmax(result, axis=-1)
        # result = result.argmax(axis=-1)
        return result

    def predict_one(self, doc):
        result = self.model.predict(doc)
        return np.argmax(result)

    def evaluate(self, testDocs, batch_size=10):
        result = self.model.evaluate_generator(testDocs)
        result = {self.model.metrics_names[i] : result[i] for i in range(len(result)) }
        return result

    def predict_one_array(self,doc):
        result = self.model.predict(doc)
        return result

    def save(self, filename):
        self.model.save(filename)

    @staticmethod
    def load_model(model_path):
        return KerasAdapter(load_model(model_path))

    def predict_generator(self, generator):
        """

        :param generator:
        :return: class ModelEvaluation
        """
        predicted = []
        trueClasses = []
        files = []
        for i in range(len(generator)):
            sys.stderr.write('\rdone {0:%}'.format(i / len(generator)))
            data = generator[i]
            r = self.predict(data[0])
            r = r.flatten()
            predicted.extend(r)
            trueClasses.extend(data[1])
            files.extend(generator.batches[i])
        return ModelEvaluation(self.model, files, trueClasses, predicted)


class KerasAutoencoderAdapter(ModelAdapter):

    def __init__(self, encoder, decoder, vae):
        self.encoder = encoder
        self.decoder = decoder
        self.vae = vae

    def fit(self, dataGenerator, epochs=1, batch_size=10, workers=2, validationDataGenerator=None,
            validationSteps=None, callbacks=None, steps_per_epoch=None):
        self.vae.fit_generator(dataGenerator, steps_per_epoch=steps_per_epoch, epochs=epochs, initial_epoch=0, max_queue_size=1, verbose=1,
                                 workers=workers, validation_data=validationDataGenerator,
                                 validation_steps=validationSteps,
                                 callbacks=callbacks)

    def save(self, filename):
        self.encoder.save('encoder_'+filename)
        self.decoder.save('decoder_' + filename)
        self.vae.save(filename)

    def predict(self, testDocs, batch_size=10):
        pass

    def predict_one(self, doc):
        pass

    def evaluate(self, testDocs, batch_size=10):
        pass

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

class Word2VecTrainer(object):
    """
    Perform training and save gensim word2vec
    """

    def __init__(self, min_count=2, size=200, workers=4, window=3, iter=10):
        self.min_count = min_count
        self.size = size
        self.workers = workers
        self.window = window
        self.iter = iter
        self.model = None

    def train(self, corpus, sg=1):
        self.model = Word2Vec(corpus, min_count=self.min_count, size=self.size, workers=self.workers, window=self.window, iter=self.iter, sg=sg)

    def save(self, filename):
        self.model.save(filename)

    def get_model(self):
        return self.model

    def load_model(self, filename):
        return Word2Vec.load(filename)

    # def load_google_model(self, filename):
    #     return KeyedVectors.load_word2vec_format(filename, binary=True)

    def retrain(self, model, corpus, sg=0):
        for i in range(0, self.iter):
            model.train(corpus, total_examples=model.corpus_count)
        self.model = model

class Doc2VecTrainer(object):
    """
        Perform training and save gensim doc2vec
        """

    def __init__(self, min_count=2, size=200, workers=4, window=3, iter=10):
        self.min_count = min_count
        self.size = size
        self.workers = workers
        self.window = window
        self.iter = iter
        self.model = None

    def train(self, corpus, sg=1):
        self.model = Doc2Vec(corpus, min_count=self.min_count, vector_size=self.size, workers=self.workers,
                              window=self.window, epochs=self.iter)

    def save(self, filename):
        self.model.save(filename)

    def get_model(self):
        return self.model

    def load_model(self, filename):
        return Doc2Vec.load(filename)

    # def load_google_model(self, filename):
    #     return KeyedVectors.load_word2vec_format(filename, binary=True)

    def retrain(self, model, corpus, sg=0):
        for i in range(0, self.iter):
            model.train(corpus, total_examples=model.corpus_count)
        self.model = model
