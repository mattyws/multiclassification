import os

from keras.engine.saving import load_model
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier
from sklearn.utils import resample

import functions
from adapter import KerasAdapter
from data_generators import LengthLongitudinalDataGenerator
from functions import print_with_time


class TrainEnsembleAdaBoosting():
    def __init__(self, data, classes, model_build_fn, epochs=100, batch_size=10, verbose=0, n_estimators=15):
        self.data = data
        self.classes = classes
        self.build_fn = model_build_fn
        self.keras_adapter = KerasClassifier(build_fn=model_build_fn, epochs=epochs, batch_size=batch_size,
                                             verbose=verbose)
        self.ensemble_classifier = AdaBoostClassifier(base_estimator=self.keras_adapter, n_estimators=n_estimators)

    def fit(self):
        #TODO: check generator use
        self.ensemble_classifier.fit(self.data, self.classes)


    def get_classifiers(self):
        return self.ensemble_classifier.estimators_


class TrainEnsembleBagging():

    def __init__(self, data, classes, model_creator, n_estimators=15, batch_size=50):
        self.data = data
        self.classes = classes
        self.model_creator = model_creator
        self.n_estimators = n_estimators
        self.batch_size = batch_size
        self.trained_estimators = 0
        self.__classifiers = []
        self.__training_data_samples = []
        self.__training_classes_samples = []
        self.__testing_data_samples = []
        self.__testing_classes_samples = []

    def fit(self, epochs=10, saved_model_prefix="bagging_{}.model"):
        indexes = [i for i in range(len(self.data))]
        for n in range(self.trained_estimators, self.n_estimators):
            print_with_time("Estimator {} of {}".format(n, self.n_estimators))
            train_indexes = resample(indexes, replace=True, n_samples=int(len(self.data) * .4))
            test_indexes = [i for i in indexes if i not in train_indexes]
            train_samples = self.data[train_indexes]
            train_classes = self.classes[train_indexes]
            test_samples = self.data[test_indexes]
            test_classes = self.classes[test_indexes]
            data_train_generator = self.__create_generator(train_samples, train_classes)
            adapter = self.model_creator.create()
            adapter.fit(data_train_generator, epochs=epochs)
            adapter.save(saved_model_prefix.format(n))
            self.__classifiers.append(saved_model_prefix.format(n))
            self.__training_data_samples.append(train_samples)
            self.__training_classes_samples.append(train_samples)
            self.__testing_data_samples.append(test_samples)
            self.__testing_classes_samples.append(test_classes)
            self.trained_estimators += 1

    def __create_generator(self, data, classes):
        train_sizes, train_labels = functions.divide_by_events_lenght(data, classes)
        data_generator = LengthLongitudinalDataGenerator(train_sizes, train_labels, max_batch_size=self.batch_size)
        data_generator.create_batches()
        return data_generator

    def get_classifiers(self):
        classifiers = []
        for classifier in self.__classifiers:
            adapter = KerasAdapter.load_model(classifier)
            model = adapter.model
            classifiers.append(model)
        return classifiers
