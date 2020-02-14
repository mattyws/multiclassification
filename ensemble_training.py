import os
import pickle

from keras.engine.saving import load_model
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier
from sklearn.utils import resample

import functions
from adapter import KerasAdapter
from data_generators import LengthLongitudinalDataGenerator, AutoencoderDataGenerator
from functions import print_with_time
from model_creators import KerasVariationalAutoencoder


def split_classes(classes):
    positive_indexes = []
    negative_indexes = []
    for index in range(len(classes)):
        if classes[index] == 0:
            negative_indexes.append(index)
        else:
            positive_indexes.append(index)
    return positive_indexes, negative_indexes



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

    def __init__(self):
        self.classifiers = []
        self.training_data_samples = []
        self.training_classes_samples = []


    def fit(self, data, classes, model_creator, training_data_samples=None, training_classes_samples=None, split_rate=.2,
            epochs=10, n_estimators=10, batch_size=30, saved_model_path="bagging_{}.model",
            saved_data_samples_path="bagging_samples_{}.model"):
        positive_indexes, negative_indexes = split_classes(classes)
        indexes = negative_indexes
        for n in range(n_estimators):
            print_with_time("Estimator {} of {}".format(n+1, n_estimators))
            if os.path.exists(saved_model_path.format(n)):
                with open(saved_data_samples_path.format(n), 'rb') as file_handler:
                    obj = pickle.load(file_handler)
                    train_samples = obj[0]
                    train_classes = obj[1]
            else:
                if training_data_samples is not None and training_classes_samples is None:
                    raise ValueError("Give the samples classes")
                elif training_data_samples is not None and training_classes_samples is not None:
                    train_samples = training_data_samples[n]
                    train_classes = training_classes_samples[n]
                else:
                    train_indexes = resample(indexes, replace=False, n_samples=int(len(positive_indexes) * split_rate))
                    train_indexes.extend(positive_indexes)
                    train_samples = data[train_indexes]
                    train_classes = classes[train_indexes]
                print(train_samples, train_classes)
                print(len(train_samples), len(train_classes))
                print(len(positive_indexes), len(negative_indexes))
                data_train_generator = self.__create_generator(train_samples, train_classes, batch_size)
                adapter = model_creator.create()
                adapter.fit(data_train_generator, epochs=epochs, use_multiprocessing=False)
                adapter.save(saved_model_path.format(n))
            with open(saved_data_samples_path.format(n), 'wb') as file_handler:
                pickle.dump((train_samples, train_classes), file_handler)
            self.classifiers.append(saved_model_path.format(n))
            self.training_data_samples.append(train_samples)
            self.training_classes_samples.append(train_samples)

    def __create_generator(self, data, classes, batch_size):
        train_sizes, train_labels = functions.divide_by_events_lenght(data, classes)
        data_generator = LengthLongitudinalDataGenerator(train_sizes, train_labels, max_batch_size=batch_size)
        data_generator.create_batches()
        return data_generator

    def get_classifiers(self):
        classifiers = []
        for classifier in self.classifiers:
            adapter = KerasAdapter.load_model(classifier)
            classifiers.append(adapter)
        return classifiers


class TrainEnsembleClustering():

    def __init__(self):
        self.classifiers = []
        self.training_data_samples = []
        self.training_classes_samples = []

    def cluster(self, data, n_cluster):
        pass

    def fit(self, data, classes, model_creator, autoencoder_creator, input_shape, n_estimators=10,
            autoencoder_batch_size=30, autoencoder_epochs=30):
        print_with_time("Training autoencoder for clustering")


    def __create_generator(self, data, classes, batch_size):
        train_sizes, train_labels = functions.divide_by_events_lenght(data, classes)
        data_generator = LengthLongitudinalDataGenerator(train_sizes, train_labels, max_batch_size=batch_size)
        data_generator.create_batches()
        return data_generator


