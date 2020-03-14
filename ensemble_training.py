import os
import pickle
from abc import abstractmethod

import multiprocessing
from functools import partial
from random import random

import numpy
import sys

from keras.engine.saving import load_model
from keras.wrappers.scikit_learn import KerasClassifier
from pyclustering.cluster import kmedoids
from sklearn.cluster.hierarchical import AgglomerativeClustering
from sklearn.cluster.k_means_ import MiniBatchKMeans
from sklearn.cluster.optics_ import OPTICS
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier
from sklearn.utils import resample
from tslearn.metrics import dtw

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

def compute_dtw_distance(frac_data, all_data=None, manager_queue=None):
    distances = dict()
    for data in frac_data:
        if manager_queue is not None:
            manager_queue.put(data)
        data_distances = []
        with open(data, 'rb') as file_handler:
            data_values = pickle.load(file_handler)
        for aux_data in all_data:
            with open(aux_data, 'rb') as aux_handler:
                aux_values = pickle.load(aux_handler)
            distance = dtw(data_values, aux_values)
            data_distances.append(distance)
        distances[data] = data_distances
    return distances

class TrainEnsembleClustering():

    def __init__(self):
        self.classifiers = []

    def generate_distance_matrix(self, data):
        with multiprocessing.Pool(processes=1) as pool:
            manager = multiprocessing.Manager()
            manager_queue = manager.Queue()
            partial_transform_representation = partial(compute_dtw_distance,
                                                       all_data=data,
                                                       manager_queue=manager_queue)
            dataset = numpy.array_split(data, 6)
            total_files = len(data)
            map_obj = pool.map_async(partial_transform_representation, dataset)
            consumed = 0
            while not map_obj.ready() or manager_queue.qsize() != 0:
                for _ in range(manager_queue.qsize()):
                    manager_queue.get()
                    consumed += 1
                sys.stderr.write('\rdone {0:%}'.format(consumed / total_files))
            result = map_obj.get()
            paths = dict()
            for r in result:
                paths.update(r)
            distance_matrix = []
            for d in data:
                distance_matrix.append(paths[d])
            return distance_matrix

    def cluster(self, data, classes, distance_matrix, n_clusters):
        positive_indexes, negative_indexes = split_classes(classes)
        print(len(positive_indexes))
        # loaded_data = self.__load_encoded_data(data[negative_indexes])
        # print_with_time("Generating distance matrix")
        # loaded_data = self.generate_distance_matrix(data[negative_indexes])
        # km = OPTICS(n_jobs=-1, cluster_method="dbscan", metric="precomputed", eps=10.0)
        # print_with_time("Training OPTICS")
        # km.fit(distance_matrix)
        # clusters_indexes = km.labels_
        print_with_time("Training K-medoids")
        initial_medoids = []
        for _ in range(n_clusters):
            initial_medoids.append(random())
        km = kmedoids.kmedoids(distance_matrix, initial_medoids, data_type='distance_matrix')
        km.process()
        clusters_indexes = km.get_clusters()
        print_with_time(clusters_indexes)
        data_samples_dict = dict()
        classes_samples_dict = dict()
        for index, cluster_index in enumerate(clusters_indexes):
            if cluster_index not in data_samples_dict.keys():
                data_samples_dict[cluster_index] = []
            data_samples_dict[cluster_index].append(data[negative_indexes[index]])
            if cluster_index not in classes_samples_dict.keys():
                classes_samples_dict[cluster_index] = []
            classes_samples_dict[cluster_index].append(0)
        print(data_samples_dict.keys())
        for key in data_samples_dict.keys():
            print(key, len(data_samples_dict[key]))
        exit()
        data_samples = []
        classes_samples = []
        for key in data_samples_dict.keys():
            samples = data_samples_dict[key]
            samples_classes = classes_samples_dict[key]
            samples.extend(data[positive_indexes])
            samples_classes.extend(classes[positive_indexes])
            data_samples.append(samples)
            classes_samples.append(samples_classes)
        return km, data_samples, classes_samples

    def fit(self, data_samples, classes_samples, model_creator, epochs=10, batch_size=30, saved_model_path="ensemble_{}.model",
            saved_data_samples_path="ensemble_samples_{}.model" ):
        n = 0
        trained_classifiers_path = []
        for data_sample, class_samples in zip(data_samples, classes_samples):
            data_train_generator = self.__create_generator(data_sample, class_samples, batch_size)
            adapter = model_creator.create()
            adapter.fit(data_train_generator, epochs=epochs, use_multiprocessing=False)
            adapter.save(saved_model_path.format(n))
            trained_classifiers_path.append(saved_model_path.format(n))
            n += 1
        return trained_classifiers_path


    def __create_generator(self, data, classes, batch_size):
        train_sizes, train_labels = functions.divide_by_events_lenght(data, classes)
        data_generator = LengthLongitudinalDataGenerator(train_sizes, train_labels, max_batch_size=batch_size)
        data_generator.create_batches()
        return data_generator

    def __load_encoded_data(self, data):
        encoded_data = []
        for d in data:
            with open(d, 'rb') as file_handler:
                encoded_data.append(numpy.array(pickle.load(file_handler)))
        return encoded_data

    @abstractmethod
    def get_classifiers(self, classifiers_path):
        classifiers = []
        for classifier in classifiers_path:
            adapter = KerasAdapter.load_model(classifier)
            classifiers.append(adapter)
        return classifiers


