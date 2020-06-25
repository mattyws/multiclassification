import csv
import datetime
import json
import os
import pickle

import pandas
import pandas as pd
import numpy as np

import keras
import sys
from keras.regularizers import l1_l2

from sklearn.model_selection._split import StratifiedKFold, train_test_split
from sklearn.utils import class_weight
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l1

import functions
from adapter import KerasAdapter
from data_generators import LengthLongitudinalDataGenerator, LongitudinalDataGenerator, MetaLearnerDataGenerator, \
    ArrayDataGenerator
from data_representation import EnsembleMetaLearnerDataCreator, TransformClinicalTextsRepresentations
from ensemble_training import TrainEnsembleAdaBoosting, TrainEnsembleBagging, split_classes
from functions import test_model, print_with_time, escape_invalid_xml_characters, escape_html_special_entities, \
    text_to_lower, remove_sepsis_mentions, remove_only_special_characters_tokens, whitespace_tokenize_text, \
    train_representation_model
from keras_callbacks import Metrics
from model_creators import MultilayerKerasRecurrentNNCreator, EnsembleModelCreator, \
    MultilayerTemporalConvolutionalNNCreator, NoteeventsClassificationModelCreator
from normalization import Normalization, NormalizationValues

def change_weak_classifiers(model):
    new_model = Model(inputs=model.input, outputs=model.layers[-2].output)
    new_model.compile(loss=model.loss, optimizer=model.optimizer)
    return new_model

def train_meta_model_on_data(data, classes, parameters):
    meta_data_input_shape = (len(data[0]),)
    modelCreator = EnsembleModelCreator(meta_data_input_shape, parameters['meta_learner_num_output_neurons'],
                                        output_units=parameters['meta_learner_output_units'],
                                        loss=parameters['meta_learner_loss'],
                                        layers_activation=parameters['meta_learner_layers_activations'],
                                        network_activation=parameters['meta_learner_network_activation'],
                                        use_dropout=parameters['meta_learner_use_dropout'],
                                        dropout=parameters['meta_learner_dropout'],
                                        # kernel_regularizer=l1_l2(l1=0.001, l2=0.01),
                                        metrics=[keras.metrics.binary_accuracy],
                                        optimizer=parameters['meta_learner_optimizer'])
    kerasAdapter = modelCreator.create()
    epochs = parameters['meta_learner_training_epochs']
    start = datetime.datetime.now()
    class_weights = class_weight.compute_class_weight('balanced',
                                                      np.unique(classes),
                                                      classes)
    mapped_weights = dict()
    for value in np.unique(classes):
        mapped_weights[value] = class_weights[value]
    class_weights = mapped_weights
    dataGenerator = ArrayDataGenerator(data, classes, parameters['meta_learner_batch_size'])
    kerasAdapter.fit(dataGenerator, epochs=epochs, use_multiprocessing=False, class_weights=class_weights)
    return kerasAdapter

def test_meta_model_on_data(model, data, classes, parameters):
    dataGenerator = ArrayDataGenerator(data, classes, parameters['meta_learner_batch_size'])
    result = test_model(model, dataGenerator, -1)
    return result

def train_evaluation_split_with_icustay(icustays, paths):
    train = []
    evaluation = []
    for path in paths:
        path_icustay = int(path.split('/')[-1].split('.')[0])
        if path_icustay in icustays:
            train.append(path)
        else:
            evaluation.append(path)
    return np.asarray(train), np.asarray(evaluation)


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
DATETIME_PATTERN = "%Y-%m-%d %H:%M:%S"
from parameters.classification_ensemble_parameters_stacking import parameters


if not os.path.exists(parameters['training_directory_path']):
    os.mkdir(parameters['training_directory_path'])

if not os.path.exists(parameters['training_directory_path'] + parameters['checkpoint']):
    os.mkdir(parameters['training_directory_path'] + parameters['checkpoint'])

with open(parameters['training_directory_path'] + parameters['checkpoint'] + "parameters.pkl", 'wb') as handler:
    pickle.dump(parameters, handler)

# Loading csv
print_with_time("Loading data")
data_csv = pd.read_csv(parameters['dataset_csv_file_path'])
data_csv = data_csv.sort_values(['icustay_id'])

# If script is using structured data, do the preparation for it (normalization and get input shape)
structured_data = None
normalization_values = None
if parameters['use_structured_data']:
    print_with_time("Preparing structured data")
    structured_data = np.array([parameters['structured_data_path'] + '{}.csv'.format(itemid) for itemid in list(data_csv['icustay_id'])])
    print_with_time("Preparing normalization values")
    normalization_values = NormalizationValues(structured_data,
                                               pickle_object_path=parameters['training_directory_path']
                                                                  + parameters['normalization_value_counts_dir'])
    normalization_values.prepare()
    # Get input shape
    aux = pd.read_csv(structured_data[0])
    if 'Unnamed: 0' in aux.columns:
        aux = aux.drop(columns=['Unnamed: 0'])
    if 'chartevents_Unnamed: 0' in aux.columns:
        aux = aux.drop(columns=['chartevents_Unnamed: 0'])
    if 'labevents_Unnamed: 0' in aux.columns:
        aux = aux.drop(columns=['labevents_Unnamed: 0'])
    if 'starttime' in aux.columns:
        aux = aux.drop(columns=['starttime'])
    if 'endtime' in aux.columns:
        aux = aux.drop(columns=['endtime'])
    structured_input_shape = (None, len(aux.columns))

# If script is using textual data, do the preparations (train word2vec)
textual_data = None
textual_transformed_data = None
if parameters['use_textual_data']:
    print_with_time("Preparing textual data")
    textual_data = np.array([parameters['textual_data_path'] + '{}.csv'.format(itemid) for itemid in list(data_csv['icustay_id'])])
    # word2vec_data = np.array([parameters['notes_word2vec_path'] + '{}.txt'.format(itemid) for itemid in textual_data])
    embedding_size = parameters['textual_embedding_size']
    min_count = parameters['textual_min_count']
    workers = parameters['textual_workers']
    window = parameters['textual_window']
    iterations = parameters['textual_iterations']
    textual_input_shape = (None, embedding_size)

    print_with_time("Training/Loading representation model")
    preprocessing_pipeline = [escape_invalid_xml_characters, escape_html_special_entities, text_to_lower,
                              whitespace_tokenize_text, remove_only_special_characters_tokens, remove_sepsis_mentions]
    representation_model = train_representation_model(textual_data,
                                                      parameters['textual_representation_model_path'],
                                                      min_count, embedding_size, workers, window, iterations,
                                                      preprocessing_pipeline=preprocessing_pipeline, word2vec=False)
    print_with_time("Transforming/Retrieving representation")
    texts_transformer = TransformClinicalTextsRepresentations(representation_model, embedding_size=embedding_size,
                                                              window=window, texts_path=parameters['textual_data_path'],
                                                              representation_save_path=parameters[
                                                                  'notes_textual_representation_path'],
                                                              is_word2vec=False)
    representation_model = None
    texts_transformer.transform(textual_data, preprocessing_pipeline=preprocessing_pipeline)
    textual_transformed_data = np.array(texts_transformer.get_new_paths(textual_data))

structured_model_creator = []
structured_model_creator.append(MultilayerKerasRecurrentNNCreator(structured_input_shape, parameters['structured_output_units'],
                                                         parameters['structured_output_neurons'],
                                                         loss=parameters['structured_loss'],
                                                         layersActivations=parameters['structured_layers_activations'],
                                                         networkActivation=parameters['structured_network_activation'],
                                                         gru=parameters['structured_gru'],
                                                          kernel_regularizer=l1(0.001),
                                                         use_dropout=parameters['structured_use_dropout'],
                                                        dropout=parameters['structured_dropout'],
                                                         metrics=[keras.metrics.binary_accuracy],
                                                         optimizer=parameters['structured_optimizer']))
structured_model_creator.append(MultilayerTemporalConvolutionalNNCreator(structured_input_shape,
                                                        parameters['structured_output_units'],
                                                        parameters['structured_output_neurons'],
                                                        loss=parameters['structured_loss'],
                                                        layersActivations=parameters[
                                                            'structured_layers_activations'],
                                                        networkActivation=parameters[
                                                            'structured_network_activation'],
                                                        pooling=parameters['structured_pooling'],
                                                        kernel_sizes=parameters['structured_kernel_sizes'],
                                                        use_dropout=parameters['structured_use_dropout'],
                                                        dilations=parameters['structured_dilations'],
                                                        nb_stacks=parameters['structured_nb_stacks'],
                                                        dropout=parameters['structured_dropout'],
                                                        kernel_regularizer=l1(0.001),
                                                        metrics=[keras.metrics.binary_accuracy],
                                                        optimizer=parameters['structured_optimizer']))
textual_model_creator = MultilayerTemporalConvolutionalNNCreator(textual_input_shape, parameters['textual_output_units'],
                                                        parameters['textual_output_neurons'],
                                                        loss=parameters['textual_loss'],
                                                        layersActivations=parameters['textual_layers_activations'],
                                                        networkActivation=parameters['textual_network_activation'],
                                                        pooling=parameters['textual_pooling'],
                                                        dilations=parameters['textual_dilations'],
                                                        nb_stacks=parameters['textual_nb_stacks'],
                                                        kernel_sizes=parameters['textual_kernel_sizes'],
                                                        kernel_regularizer=l1(0.01),
                                                        use_dropout=parameters['textual_use_dropout'],
                                                        dropout=parameters['textual_dropout'],
                                                        metrics=[keras.metrics.binary_accuracy],
                                                        optimizer=parameters['textual_optimizer'])
# Using a seed always will get the same data split even if the training stops
print_with_time("Transforming classes")
classes = np.array([1 if c == 'sepsis' else 0 for c in list(data_csv['class'])])

print_with_time("Training/evaluation data spliting")
icustays = np.asarray(data_csv['icustay_id'].tolist())

if not os.path.exists(parameters['training_directory_path'] + parameters['checkpoint'] + parameters['train_icustays_samples']):
    X, X_val, classes, classes_evaluation = train_test_split(icustays, classes, test_size=0.20)
    with open(parameters['training_directory_path'] + parameters['checkpoint'] + parameters['train_icustays_samples'], 'wb') \
        as f:
        pickle.dump(X, f)
    with open(parameters['training_directory_path'] + parameters['checkpoint'] + parameters['train_icustays_classes'], 'wb') \
        as f:
        pickle.dump(classes, f)
    with open(parameters['training_directory_path'] + parameters['checkpoint'] + parameters['test_icustays_samples'], 'wb') \
        as f:
        pickle.dump(X_val, f)
    with open(parameters['training_directory_path'] + parameters['checkpoint'] + parameters['test_icustays_classes'], 'wb') \
        as f:
        pickle.dump(classes_evaluation, f)
else:
    with open(parameters['training_directory_path'] + parameters['checkpoint'] + parameters['train_icustays_samples'], 'rb') \
        as f:
        X = pickle.load(f)
    with open(parameters['training_directory_path'] + parameters['checkpoint'] + parameters['train_icustays_classes'], 'rb') \
        as f:
        classes = pickle.load(f)
    with open(parameters['training_directory_path'] + parameters['checkpoint'] + parameters['test_icustays_samples'], 'rb') \
        as f:
        X_val = pickle.load(f)
    with open(parameters['training_directory_path'] + parameters['checkpoint'] + parameters['test_icustays_classes'], 'rb') \
        as f:
        classes_evaluation = pickle.load(f)
structured_data, structured_evaluation = train_evaluation_split_with_icustay(X, structured_data)
textual_transformed_data, textual_evaluation = train_evaluation_split_with_icustay(X, textual_transformed_data)
print(len(structured_data), len(classes))
print(len(structured_evaluation), len(classes_evaluation))
print(len(textual_transformed_data), len(classes))
print(len(textual_evaluation), len(classes_evaluation))
# exit()

kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=15)
fold = 0
structured_predictions = None
structured_representations = None
textual_predictions = None
textual_representations = None

all_predictions = None
all_representations = None
all_metrics = None
# ====================== Script that start training new models
with open(parameters['training_directory_path'] + parameters['checkpoint'] + parameters['results_file_name'], 'a+') as cvsFileHandler, \
        open(parameters['training_directory_path'] + parameters['checkpoint'] + parameters['level_zero_result_file_name'], 'a+')\
                as level_zero_csv_file_handler: # where the results for each fold are appended
    dictWriter = None
    level_zero_dict_writer = None
    for trainIndex, testIndex in kf.split(structured_data, classes):
        print(len(trainIndex), len(testIndex))
        print(len(structured_data[trainIndex]), len(structured_data[testIndex]))
        print(len(textual_transformed_data[trainIndex]), len(textual_transformed_data[testIndex]))
        print(len(classes[trainIndex]), len(classes[testIndex]))
        jump_fold = True
        if os.path.exists(parameters['training_directory_path'] + parameters['checkpoint']
                                + parameters['fold_metrics_file_csv'].format(fold)):
            print("Pass fold {}".format(fold))

            fold_structured_predictions = pandas.read_csv(parameters['training_directory_path'] + parameters['checkpoint']
                                               + parameters['fold_structured_predictions_file_csv'].format(fold))
            fold_structured_representations= pandas.read_csv(parameters['training_directory_path'] + parameters['checkpoint']
                                                   + parameters['fold_structured_representations_file_csv'].format(
                fold))
            fold_textual_predictions= pandas.read_csv(parameters['training_directory_path'] + parameters['checkpoint']
                                            + parameters['fold_textual_predictions_file_csv'].format(fold))
            fold_textual_representations= pandas.read_csv(parameters['training_directory_path'] + parameters['checkpoint']
                                                + parameters['fold_textual_representations_file_csv'].format(fold))

            fold_metrics = pandas.read_csv(parameters['training_directory_path'] + parameters['checkpoint']
                                + parameters['fold_metrics_file_csv'].format(fold))

            if all_metrics is None:
                structured_predictions = fold_structured_predictions
                structured_representations = fold_structured_representations
                textual_predictions = fold_textual_predictions
                textual_representations = fold_textual_representations
                all_metrics = fold_metrics
            else:
                structured_predictions = structured_predictions.append(fold_structured_predictions)
                structured_representations = structured_representations.append(fold_structured_representations)
                textual_predictions = textual_predictions.append(fold_textual_predictions)
                textual_representations = textual_representations.append(fold_textual_representations)
                all_metrics = all_metrics.append(fold_metrics, ignore_index=True)
            fold += 1
            continue
        print_with_time("Fold {}".format(fold))

        class_weights = class_weight.compute_class_weight('balanced',
                                                          np.unique(classes[trainIndex]),
                                                          classes[trainIndex])
        mapped_weights = dict()
        for value in np.unique(classes):
            mapped_weights[value] = class_weights[value]
        class_weights = mapped_weights
        fold_structured_predictions = dict()
        fold_structured_representations = dict()
        fold_textual_predictions = dict()
        fold_textual_representations = dict()

        fold_predictions = dict()
        fold_representations = dict()
        fold_metrics = []

        structured_ensemble = None
        model_adapters = []
        print_with_time("Getting values for normalization")
        values = normalization_values.get_normalization_values(structured_data[trainIndex],
                                                               saved_file_name=parameters['training_directory_path']
                                                                               + parameters['normalization_data_path'].format(fold))
        normalizer = Normalization(values, temporary_path=parameters['training_directory_path']
                                                          + parameters['normalized_structured_data_path'].format(fold))

        print_with_time("Normalizing fold data")
        normalizer.normalize_files(structured_data)
        normalized_data = np.array(normalizer.get_new_paths(structured_data))

        train_sizes, train_labels = functions.divide_by_events_lenght(normalized_data[trainIndex]
                              , classes[trainIndex]
                              , sizes_filename=parameters['training_directory_path'] +
                                               parameters['structured_training_events_sizes_file'].format(fold)
                              , classes_filename=parameters['training_directory_path'] +
                                                 parameters['structured_training_events_sizes_labels_file'].format(fold))
        test_sizes, test_labels = functions.divide_by_events_lenght(normalized_data[testIndex], classes[testIndex]
                                    , sizes_filename=parameters['training_directory_path'] +
                                                     parameters['structured_testing_events_sizes_file'].format(fold)
                                    , classes_filename=parameters['training_directory_path'] +
                                                       parameters['structured_testing_events_sizes_labels_file'].format(fold))
        dataTrainGenerator = LengthLongitudinalDataGenerator(train_sizes, train_labels,
                                                             max_batch_size=parameters['structured_batch_size'])
        dataTrainGenerator.create_batches()
        dataTestGenerator = LengthLongitudinalDataGenerator(test_sizes, test_labels,
                                                            max_batch_size=parameters['structured_batch_size'])
        dataTestGenerator.create_batches()


        start = datetime.datetime.now()
        print_with_time("Training level 0 models for structured data")
        for i, model_creator in enumerate(structured_model_creator):
            adapter = model_creator.create()
            adapter.fit(dataTrainGenerator, epochs=parameters['structured_training_epochs'], callbacks=None,
                        class_weights=class_weights, use_multiprocessing=False)
            metrics, results = test_model(adapter, dataTestGenerator, fold, return_predictions=True)
            metrics['model'] = model_creator.name
            fold_metrics.append(metrics)
            for key in results.keys():
                if not model_creator.name in fold_structured_predictions.keys():
                    fold_structured_predictions[model_creator.name] = dict()
                icustay = key.split('/')[-1].split('.')[0]
                fold_structured_predictions[model_creator.name][icustay] = results[key]
            model = change_weak_classifiers(adapter.model)
            for i in range(len(dataTestGenerator)):
                sys.stderr.write('\rdone {0:%}'.format(i / len(dataTestGenerator)))
                data = dataTestGenerator[i]
                representations = model.predict(data[0])
                # print(representations)
                for f, r in zip(dataTestGenerator.batches[i], representations):
                    icustay = f.split('/')[-1].split('.')[0]
                    if icustay not in fold_structured_representations.keys():
                        fold_structured_representations[icustay] = []
                        fold_structured_representations[icustay].extend(r)


        end = datetime.datetime.now()
        time_to_train = end - start
        hours, remainder = divmod(time_to_train.seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        print_with_time('Took {:02}:{:02}:{:02} to train the level zero models for structured data'.format(int(hours), int(minutes), int(seconds)))

        train_sizes, train_labels = functions.divide_by_events_lenght(textual_transformed_data[trainIndex]
                              , classes[trainIndex]
                              , sizes_filename=parameters['training_directory_path'] +
                                                parameters['textual_training_events_sizes_file'].format(fold)
                              , classes_filename=parameters['training_directory_path'] +
                                                 parameters['textual_training_events_sizes_labels_file'].format(fold))
        test_sizes, test_labels = functions.divide_by_events_lenght(textual_transformed_data[testIndex], classes[testIndex]
                            , sizes_filename=parameters['training_directory_path'] +
                                             parameters['textual_testing_events_sizes_file'].format(fold)
                            , classes_filename=parameters['training_directory_path'] +
                                               parameters['textual_testing_events_sizes_labels_file'].format(fold))
        dataTrainGenerator = LengthLongitudinalDataGenerator(train_sizes, train_labels,
                                                             max_batch_size=parameters['structured_batch_size'])
        dataTrainGenerator.create_batches()
        dataTestGenerator = LengthLongitudinalDataGenerator(test_sizes, test_labels,
                                                            max_batch_size=parameters['structured_batch_size'])
        dataTestGenerator.create_batches()

        print_with_time("Training level 0 models for textual data")
        start = datetime.datetime.now()
        adapter = textual_model_creator.create()
        adapter.fit(dataTrainGenerator, epochs=parameters['textual_training_epochs'], callbacks=None,
                    class_weights=class_weights, use_multiprocessing=False)
        metrics, results = test_model(adapter, dataTestGenerator, fold, return_predictions=True)
        metrics['model'] = model_creator.name + "_textual"
        fold_metrics.append(metrics)
        for key in results.keys():
            if not model_creator.name + "_textual" in fold_textual_predictions.keys():
                fold_textual_predictions[model_creator.name + "_textual"] = dict()
            icustay = key.split('/')[-1].split('.')[0]
            fold_textual_predictions[model_creator.name + "_textual"][icustay] = results[key]
        model = change_weak_classifiers(adapter.model)
        for i in range(len(dataTestGenerator)):
            sys.stderr.write('\rdone {0:%}'.format(i / len(dataTestGenerator)))
            data = dataTestGenerator[i]
            representations = model.predict(data[0])
            # print(representations)
            for f, r in zip(dataTestGenerator.batches[i], representations):
                icustay = f.split('/')[-1].split('.')[0]
                if icustay not in fold_textual_representations.keys():
                    fold_textual_representations[icustay] = []
                fold_textual_representations[icustay].extend(r)

        end = datetime.datetime.now()
        time_to_train = end - start
        hours, remainder = divmod(time_to_train.seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        print_with_time(
            'Took {:02}:{:02}:{:02} to train the level zero models for textual data'.format(int(hours),
                                                                                               int(minutes),
                                                                                               int(seconds)))

        fold_structured_predictions = pandas.DataFrame(fold_structured_predictions)
        fold_structured_representations = pandas.DataFrame(fold_structured_representations).transpose()

        fold_textual_predictions = pandas.DataFrame(fold_textual_predictions)
        fold_textual_representations = pandas.DataFrame(fold_textual_representations).transpose()

        structured_representations = fold_structured_representations.add_prefix("s_")
        fold_textual_representations = fold_textual_representations.add_prefix("t_")


        fold_metrics = pandas.DataFrame(fold_metrics)

        fold_structured_predictions.to_csv(parameters['training_directory_path'] + parameters['checkpoint']
                                + parameters['fold_structured_predictions_file_csv'].format(fold))
        fold_structured_representations.to_csv(parameters['training_directory_path'] + parameters['checkpoint']
                            + parameters['fold_structured_representations_file_csv'].format(fold))
        fold_textual_predictions.to_csv(parameters['training_directory_path'] + parameters['checkpoint']
                                           + parameters['fold_textual_predictions_file_csv'].format(fold))
        fold_textual_representations.to_csv(parameters['training_directory_path'] + parameters['checkpoint']
                                               + parameters['fold_textual_representations_file_csv'].format(fold))

        fold_metrics.to_csv(parameters['training_directory_path'] + parameters['checkpoint']
                                + parameters['fold_metrics_file_csv'].format(fold))

        if all_predictions is None:
            structured_predictions = fold_structured_predictions
            structured_representations = fold_structured_representations
            textual_predictions = fold_textual_predictions
            textual_representations = fold_textual_representations

            all_metrics = fold_metrics
        else:
            structured_predictions = structured_predictions.append(fold_structured_predictions)
            structured_representations = structured_representations.append(fold_structured_representations)
            textual_predictions = textual_predictions.append(fold_textual_predictions)
            textual_representations = textual_representations.append(fold_textual_representations)

            all_metrics = all_metrics.append(fold_metrics)
        fold += 1

from scipy.stats import zscore

# TODO: check if exists
if 'Unnamed: 0' in structured_predictions:
    structured_predictions = structured_predictions.set_index(['Unnamed: 0'])
    structured_representations = structured_representations.set_index(['Unnamed: 0'])
    textual_predictions = textual_predictions.set_index(['Unnamed: 0'])
    textual_representations = textual_representations.set_index(['Unnamed: 0'])

structured_predictions.to_csv(parameters['training_directory_path'] + parameters['checkpoint']
                                + parameters['structured_predictions_file_csv'])
structured_representations.to_csv(parameters['training_directory_path'] + parameters['checkpoint']
                                + parameters['structured_representations_file_csv'])
textual_predictions.to_csv(parameters['training_directory_path'] + parameters['checkpoint']
                                + parameters['textual_predictions_file_csv'])
textual_representations.to_csv(parameters['training_directory_path'] + parameters['checkpoint']
                                + parameters['textual_representations_file_csv'])
all_metrics.to_csv(parameters['training_directory_path'] + parameters['checkpoint']
                                + parameters['metrics_file_csv'])

class_weights = class_weight.compute_class_weight('balanced',
                                                  np.unique(classes),
                                                  classes)
mapped_weights = dict()
for value in np.unique(classes):
    mapped_weights[value] = class_weights[value]
class_weights = mapped_weights


print(all_metrics)
metric_mean = all_metrics.groupby(by=['model']).mean()
print(metric_mean)

structured_evaluate_predictions = dict()
structured_evaluate_representations = dict()

print_with_time("Getting values for normalization")
values = normalization_values.get_normalization_values(structured_data,
                                                       saved_file_name=parameters['training_directory_path']
                                                                       + parameters['normalization_data_path'].format("all"))
normalizer = Normalization(values, temporary_path=parameters['training_directory_path']
                                                  + parameters['normalized_structured_data_path'].format("all"))

print_with_time("Normalizing all data")
normalizer.normalize_files(structured_data)
normalized_data = np.array(normalizer.get_new_paths(structured_data))
normalizer.normalize_files(structured_evaluation)
normalized_evaluation = np.array(normalizer.get_new_paths(structured_evaluation))
print(normalized_evaluation)
print(len(normalized_evaluation))

train_sizes, train_labels = functions.divide_by_events_lenght(normalized_data
                              , classes
                              , sizes_filename=parameters['training_directory_path'] +
                                               parameters['structured_training_events_sizes_file'].format("all")
                              , classes_filename=parameters['training_directory_path'] +
                                                 parameters['structured_training_events_sizes_labels_file'].format("all"))
test_sizes, test_labels = functions.divide_by_events_lenght(normalized_evaluation, classes_evaluation
                            , sizes_filename=parameters['training_directory_path'] +
                                             parameters['structured_testing_events_sizes_file'].format("all")
                            , classes_filename=parameters['training_directory_path'] +
                                               parameters['structured_testing_events_sizes_labels_file'].format("all"))
dataTrainGenerator = LengthLongitudinalDataGenerator(train_sizes, train_labels,
                                                     max_batch_size=parameters['structured_batch_size'])
dataTrainGenerator.create_batches()
dataTestGenerator = LengthLongitudinalDataGenerator(test_sizes, test_labels,
                                                    max_batch_size=parameters['structured_batch_size'])
dataTestGenerator.create_batches()
structured_evaluation_predictions = dict()
structured_evaluation_representations = dict()
print_with_time("Training strctured models on all data")
for i, model_creator in enumerate(structured_model_creator):
    model_creator.name = model_creator.name + "_all"
    if not os.path.exists(parameters['training_directory_path'] + parameters['checkpoint']
                                  + parameters['structured_weak_model_all'].format(model_creator.name)):
        adapter = model_creator.create()
        adapter.fit(dataTrainGenerator, epochs=parameters['structured_training_epochs'], callbacks=None,
                    class_weights=class_weights, use_multiprocessing=False)
        adapter.save(parameters['training_directory_path'] + parameters['checkpoint']
                                  + parameters['structured_weak_model_all'].format(model_creator.name))
    else:
        adapter = KerasAdapter.load_model(parameters['training_directory_path'] + parameters['checkpoint']
                                  + parameters['structured_weak_model_all'].format(model_creator.name))
    metrics, results = test_model(adapter, dataTestGenerator, -1, return_predictions=True)
    metrics['model'] = model_creator.name
    all_metrics.append(metrics, ignore_index=True)
    for key in results.keys():
        if not model_creator.name in structured_evaluation_predictions.keys():
            structured_evaluation_predictions[model_creator.name] = dict()
        icustay = key.split('/')[-1].split('.')[0]
        structured_evaluation_predictions[model_creator.name][icustay] = results[key]
    model = change_weak_classifiers(adapter.model)
    for i in range(len(dataTestGenerator)):
        sys.stderr.write('\rdone {0:%}'.format(i / len(dataTestGenerator)))
        data = dataTestGenerator[i]
        representations = model.predict(data[0])
        # print(representations)
        for f, r in zip(dataTestGenerator.batches[i], representations):
            icustay = f.split('/')[-1].split('.')[0]
            if icustay not in structured_evaluation_representations.keys():
                structured_evaluation_representations[icustay] = []
            structured_evaluation_representations[icustay].extend(r)

structured_evaluation_predictions = pandas.DataFrame(structured_evaluation_predictions)
structured_evaluation_representations = pandas.DataFrame(structured_evaluation_representations).transpose()
structured_evaluation_representations = structured_evaluation_representations.add_prefix("s_")

print_with_time("Training textual models on all data")
textual_evaluate_predictions = dict()
textual_evaluate_representations = dict()
train_sizes, train_labels = functions.divide_by_events_lenght(textual_transformed_data
                              , classes
                              , sizes_filename=parameters['training_directory_path'] +
                                                parameters['textual_training_events_sizes_file'].format("all")
                              , classes_filename=parameters['training_directory_path'] +
                                                 parameters['textual_training_events_sizes_labels_file'].format("all"))
test_sizes, test_labels = functions.divide_by_events_lenght(textual_evaluation, classes_evaluation
                    , sizes_filename=parameters['training_directory_path'] +
                                     parameters['textual_testing_events_sizes_file'].format("all")
                    , classes_filename=parameters['training_directory_path'] +
                                       parameters['textual_testing_events_sizes_labels_file'].format("all"))
dataTrainGenerator = LengthLongitudinalDataGenerator(train_sizes, train_labels,
                                                     max_batch_size=parameters['structured_batch_size'])
dataTrainGenerator.create_batches()
dataTestGenerator = LengthLongitudinalDataGenerator(test_sizes, test_labels,
                                                    max_batch_size=parameters['structured_batch_size'])
dataTestGenerator.create_batches()
start = datetime.datetime.now()
model_creator.name = model_creator.name + "_all_textual"
if not os.path.exists(parameters['training_directory_path'] + parameters['checkpoint']
                              + parameters['textual_weak_model_all'].format(model_creator.name)):
    adapter = textual_model_creator.create()
    adapter.fit(dataTrainGenerator, epochs=parameters['structured_training_epochs'], callbacks=None,
                class_weights=class_weights, use_multiprocessing=False)
    adapter.save(parameters['training_directory_path'] + parameters['checkpoint']
                              + parameters['textual_weak_model_all'].format(model_creator.name))
else:
    adapter = KerasAdapter.load_model(parameters['training_directory_path'] + parameters['checkpoint']
                                      + parameters['textual_weak_model_all'].format(model_creator.name))
metrics, results = test_model(adapter, dataTestGenerator, -1, return_predictions=True)
metrics['model'] = model_creator.name
all_metrics.append(metrics, ignore_index=True)
for key in results.keys():
    if not model_creator.name in textual_evaluate_predictions.keys():
        textual_evaluate_predictions[model_creator.name] = dict()
    icustay = key.split('/')[-1].split('.')[0]
    textual_evaluate_predictions[model_creator.name][icustay] = results[key]
model = change_weak_classifiers(adapter.model)
for i in range(len(dataTestGenerator)):
    sys.stderr.write('\rdone {0:%}'.format(i / len(dataTestGenerator)))
    data = dataTestGenerator[i]
    representations = model.predict(data[0])
    for f, r in zip(dataTestGenerator.batches[i], representations):
        icustay = f.split('/')[-1].split('.')[0]
        if icustay not in textual_evaluate_representations.keys():
            textual_evaluate_representations[icustay] = []
        textual_evaluate_representations[icustay].extend(r)

textual_evaluate_predictions = pandas.DataFrame(textual_evaluate_predictions)
textual_evaluate_representations = pandas.DataFrame(textual_evaluate_representations).transpose()
textual_evaluate_representations  = textual_evaluate_representations.add_prefix("t_")

ensemble_results = dict()

data_csv.loc[:, 'class'] = data_csv['class'].apply(lambda x: 1 if x == 'sepsis' else 0)
print(data_csv['class'])

dataset_patients = pandas.read_csv('/home/mattyws/Documents/mimic/sepsis3-df-no-exclusions.csv')
dataset_patients[['age', 'is_male', 'height', 'weight']] = dataset_patients[['age', 'is_male', 'height', 'weight']].apply(zscore)
dataset_patients[['age', 'is_male', 'gender', 'height', 'weight']] = dataset_patients[['age', 'is_male', 'gender', 'height', 'weight']].fillna(0)
meta_data_extra = dataset_patients[['icustay_id', 'age', 'is_male', 'height', 'weight']]

structured_predictions = structured_predictions.set_index(pd.to_numeric(structured_predictions.index))
structured_evaluation_predictions = structured_evaluation_predictions\
    .set_index(pd.to_numeric(structured_evaluation_predictions.index))
textual_predictions = textual_predictions.set_index(pd.to_numeric(textual_predictions.index))
textual_evaluate_predictions = textual_evaluate_predictions.set_index(pd.to_numeric(textual_evaluate_predictions.index))

structured_representations = structured_representations.set_index(pd.to_numeric(structured_representations.index))
structured_evaluation_representations = structured_evaluation_representations\
    .set_index(pd.to_numeric(structured_evaluation_representations.index))
textual_representations = textual_representations.set_index(pd.to_numeric(textual_representations.index))
textual_evaluate_representations = textual_evaluate_representations\
    .set_index(pd.to_numeric(textual_evaluate_representations.index))

print(structured_predictions)

meta_data_predictions_structured = pandas.merge(structured_predictions, meta_data_extra, left_index=True,
                                                right_on="icustay_id", how="left")
meta_evaluation_predictions_structured = pandas.merge(structured_evaluation_predictions, meta_data_extra, left_index=True,
                                                right_on="icustay_id", how="left")
print(meta_data_predictions_structured)
print(meta_data_predictions_structured.columns)
meta_data_predictions_structured = pandas.merge(meta_data_predictions_structured, data_csv[['icustay_id', 'class']],
                                                left_on="icustay_id", right_on="icustay_id", how="left")
meta_evaluation_predictions_structured = pandas.merge(meta_evaluation_predictions_structured, data_csv[['icustay_id', 'class']],
                                                left_on="icustay_id", right_on="icustay_id", how="left")
print(meta_data_predictions_structured.columns)
if 'Unnamed: 0' in meta_data_predictions_structured.columns:
    meta_data_predictions_structured = meta_data_predictions_structured.drop(columns=['Unnamed: 0'])
if 'Unnamed: 0' in meta_evaluation_predictions_structured.columns:
    meta_evaluation_predictions_structured = meta_evaluation_predictions_structured.drop(columns=['Unnamed: 0'])
columns = [c for c in meta_data_predictions_structured.columns if 'class' not in c and 'icustay' not in c]
columns_evaluation = [c for c in meta_evaluation_predictions_structured.columns if 'class' not in c and 'icustay' not in c]
print(columns)
print(columns_evaluation)
print(meta_data_predictions_structured.columns)
training_values = meta_data_predictions_structured.loc[:, columns]
training_values = training_values.values
training_classes = np.asarray(pd.to_numeric(meta_data_predictions_structured['class']).tolist())

testing_values = meta_evaluation_predictions_structured.loc[:, columns_evaluation]
testing_values = testing_values.values
testing_classes = np.asarray(pd.to_numeric(meta_evaluation_predictions_structured['class']).tolist())

meta_adapter = train_meta_model_on_data(training_values, training_classes, parameters)
result = test_meta_model_on_data(meta_adapter, testing_values, testing_classes, parameters)
ensemble_results['sp'] = result
# TODO: test model with the splited data

all_predictions = pd.merge(structured_predictions, textual_predictions, left_index=True, right_index=True)
all_evaluations = pd.merge(structured_evaluation_predictions, textual_evaluate_predictions, left_index=True, right_index=True)
meta_data_predictions = pandas.merge(all_predictions, meta_data_extra, left_index=True,
                                                right_on="icustay_id", how="left")
meta_data_evaluation = pandas.merge(all_evaluations, meta_data_extra, left_index=True,
                                                right_on="icustay_id", how="left")
meta_data_predictions = pandas.merge(meta_data_predictions, data_csv[['icustay_id', 'class']],
                                                left_on="icustay_id", right_on="icustay_id", how="left")
meta_data_evaluation = pandas.merge(meta_data_evaluation, data_csv[['icustay_id', 'class']],
                                                left_on="icustay_id", right_on="icustay_id", how="left")
print(meta_data_predictions.columns)
if 'Unnamed: 0' in meta_data_predictions.columns:
    meta_data_predictions = meta_data_predictions.drop(columns=['Unnamed: 0'])
if 'Unnamed: 0' in meta_data_evaluation.columns:
    meta_data_evaluation = meta_data_evaluation.drop(columns=['Unnamed: 0'])
columns = [c for c in meta_data_predictions.columns if 'class' not in c and 'icustay' not in c]
columns_evaluation = [c for c in meta_data_evaluation.columns if 'class' not in c and 'icustay' not in c]
training_values = meta_data_predictions.loc[:, columns]
training_values = training_values.values
training_classes = np.asarray(pd.to_numeric(meta_data_predictions['class']).tolist())

testing_values = meta_data_evaluation.loc[:, columns_evaluation]
testing_values = testing_values.values
testing_classes = np.asarray(pd.to_numeric(meta_data_evaluation['class']).tolist())

meta_adapter = train_meta_model_on_data(training_values, training_classes, parameters)
result = test_meta_model_on_data(meta_adapter, testing_values, testing_classes, parameters)
ensemble_results['bp'] = result
# TODO: test model with the splited data

# REPRESENTATIONS

structured_representations = structured_representations.set_index(pd.to_numeric(structured_representations.index))
print(structured_representations)
meta_data_representations_structured = pandas.merge(structured_representations, meta_data_extra, left_index=True,
                                                right_on="icustay_id", how="left")
meta_evaluation_representations_structured = pandas.merge(structured_evaluation_representations, meta_data_extra, left_index=True,
                                                right_on="icustay_id", how="left")
print(meta_data_representations_structured)
print(meta_data_representations_structured.columns)
meta_data_representations_structured = pandas.merge(meta_data_representations_structured, data_csv[['icustay_id', 'class']],
                                                left_on="icustay_id", right_on="icustay_id", how="left")
meta_evaluation_representations_structured = pandas.merge(meta_evaluation_representations_structured, data_csv[['icustay_id', 'class']],
                                                left_on="icustay_id", right_on="icustay_id", how="left")
print(meta_data_representations_structured.columns)
if 'Unnamed: 0' in meta_data_representations_structured.columns:
    meta_data_representations_structured = meta_data_representations_structured.drop(columns=['Unnamed: 0'])
if 'Unnamed: 0' in meta_evaluation_representations_structured.columns:
    meta_evaluation_representations_structured = meta_evaluation_representations_structured.drop(columns=['Unnamed: 0'])
columns = [c for c in meta_data_representations_structured.columns if 'class' not in c and 'icustay' not in c]
columns_evaluation = [c for c in meta_evaluation_representations_structured.columns if 'class' not in c and 'icustay' not in c]
print(columns)
print(meta_data_representations_structured.columns)
training_values = meta_data_representations_structured.loc[:, columns]
print(training_values)
training_values = training_values.values
training_classes = np.asarray(pd.to_numeric(meta_data_representations_structured['class']).tolist())
print(training_classes)

testing_values = meta_evaluation_representations_structured.loc[:, columns_evaluation]
testing_values = testing_values.values
testing_classes = np.asarray(pd.to_numeric(meta_evaluation_representations_structured['class']).tolist())

meta_adapter = train_meta_model_on_data(training_values, training_classes, parameters)
result = test_meta_model_on_data(meta_adapter, testing_values, testing_classes, parameters)
ensemble_results['sr'] = result
# TODO: test model with the splited data

all_representations = pd.merge(structured_representations, textual_representations, left_index=True, right_index=True)
all_evaluations = pd.merge(structured_evaluation_representations, textual_evaluate_representations,
                               left_index=True, right_index=True)
meta_data_representations= pandas.merge(all_representations, meta_data_extra, left_index=True,
                                                right_on="icustay_id")
meta_evaluation_representations= pandas.merge(all_evaluations, meta_data_extra, left_index=True,
                                                right_on="icustay_id")
meta_data_representations = pandas.merge(meta_data_representations, data_csv[['icustay_id', 'class']],
                                                left_on="icustay_id", right_on="icustay_id")
meta_evaluation_representations = pandas.merge(meta_evaluation_representations, data_csv[['icustay_id', 'class']],
                                                left_on="icustay_id", right_on="icustay_id")
print(meta_data_representations.columns)
if 'Unnamed: 0' in meta_data_representations.columns:
    meta_data_representations = meta_data_representations.drop(columns=['Unnamed: 0'])
if 'Unnamed: 0' in meta_evaluation_representations.columns:
    meta_evaluation_representations = meta_evaluation_representations.drop(columns=['Unnamed: 0'])
columns = [c for c in meta_data_representations.columns if 'class' not in c and 'icustay' not in c]
columns_evaluation = [c for c in meta_evaluation_representations.columns if 'class' not in c and 'icustay' not in c]
print(columns)
print(meta_data_representations.columns)
training_values = meta_data_representations.loc[:, columns]
print(training_values)
training_values = training_values.values
training_classes = np.asarray(pd.to_numeric(meta_data_representations['class']).tolist())
print(training_classes)

testing_values = meta_evaluation_representations.loc[:, columns_evaluation]
testing_values = testing_values.values
texting_classes = np.asarray(pd.to_numeric(meta_evaluation_representations['class']).tolist())

meta_adapter = train_meta_model_on_data(training_values, training_classes, parameters)

meta_adapter = train_meta_model_on_data(training_values, training_classes, parameters)
result = test_meta_model_on_data(meta_adapter, testing_values, testing_classes, parameters)
ensemble_results['br'] = result
# TODO: test model with the splited data

ensemble_results = pandas.DataFrame(ensemble_results)
ensemble_results.to_csv(parameters['training_directory_path'] + parameters['checkpoint']
                                + parameters['ensemble_results_file'])
all_metrics.to_csv(parameters['training_directory_path'] + parameters['checkpoint']
                                + parameters['metrics_file_csv'])


# print_with_time("Get model from adapters")
# aux_level_zero_models = []
# for adapter in level_zero_models:
#     if isinstance(adapter, tuple):
#         aux_level_zero_models.append((adapter[0].model, adapter[1]))
#     else:
#         aux_level_zero_models.append(adapter.model)
# level_zero_models = aux_level_zero_models
#
# print_with_time("Creating meta model data")
# meta_data_creator = EnsembleMetaLearnerDataCreator(level_zero_models,
#                                                    use_class_prediction=parameters['use_class_prediction'])
# meta_data_creator.create_meta_learner_data(meta_data, parameters['training_directory_path']
#                                            + parameters['checkpoint']
#                                            + parameters['meta_representation_path'].format(fold))
#
# meta_data = np.array(meta_data_creator.get_new_paths(meta_data))
# representation_chunk_size = 0
# if parameters['use_structured_data'] and parameters['use_textual_data']:
#     representation_chunk_size = 2 if parameters['use_class_prediction'] else parameters['structured_output_units'][-1] \
#                                 + parameters['textual_output_units'][-1]
# elif parameters['use_structured_data']:
#     representation_chunk_size = 1 if parameters['use_class_prediction'] else parameters['structured_output_units'][-1]
# elif parameters['use_textual_data']:
#     representation_chunk_size = 1 if parameters['use_class_prediction'] else parameters['textual_output_units'][-1]
#
# for num_models in range(1, parameters['n_estimators']+1):
#
#     print_with_time("Creating meta data generators")
#     training_meta_data_generator = MetaLearnerDataGenerator(meta_data[trainIndex], classes[trainIndex],
#                                                             parameters['meta_learner_batch_size'],
#                                                             num_models,
#                                                             representation_chunk_size)
#     testing_meta_data_generator = MetaLearnerDataGenerator(meta_data[testIndex], classes[testIndex],
#                                                             parameters['meta_learner_batch_size'],
#                                                            num_models,
#                                                            representation_chunk_size)
#
#     meta_data_input_shape = (num_models * representation_chunk_size, )
#     modelCreator = EnsembleModelCreator(meta_data_input_shape, parameters['meta_learner_num_output_neurons'],
#                                         output_units=parameters['meta_learner_output_units'],
#                                         loss=parameters['meta_learner_loss'],
#                                         layers_activation=parameters['meta_learner_layers_activations'],
#                                         network_activation=parameters['meta_learner_network_activation'],
#                                         use_dropout=parameters['meta_learner_use_dropout'],
#                                         dropout=parameters['meta_learner_dropout'],
#                                         # kernel_regularizer=l1_l2(l1=0.001, l2=0.01),
#                                         metrics=[keras.metrics.binary_accuracy],
#                                         optimizer=parameters['meta_learner_optimizer'])
#     kerasAdapter = modelCreator.create()
#     epochs = parameters['meta_learner_training_epochs']
#     print_with_time("Training model with {} models".format(num_models))
#     start = datetime.datetime.now()
#     class_weights = class_weight.compute_class_weight('balanced',
#                                                       np.unique(classes[trainIndex]),
#                                                       classes[trainIndex])
#     mapped_weights = dict()
#     for value in np.unique(classes):
#         mapped_weights[value] = class_weights[value]
#     class_weights = mapped_weights
#     kerasAdapter.fit(training_meta_data_generator, epochs=epochs, use_multiprocessing=False, class_weights=class_weights)
#     end = datetime.datetime.now()
#     time_to_train = end - start
#     hours, remainder = divmod(time_to_train.seconds, 3600)
#     minutes, seconds = divmod(remainder, 60)
#     print_with_time('Took {:02}:{:02}:{:02} to train the model'.format(int(hours), int(minutes), int(seconds)))
#     print_with_time("Testing model")
#     metrics = test_model(kerasAdapter, testing_meta_data_generator, fold)
#     metrics['num_models'] = num_models
#     if dictWriter is None:
#         dictWriter = csv.DictWriter(cvsFileHandler, metrics.keys())
#     if fold == 0 and num_models == 1:
#         dictWriter.writeheader()
#     dictWriter.writerow(metrics)
#     kerasAdapter.save(parameters['training_directory_path'] + parameters['checkpoint']
#                       + parameters['meta_model_file_name'].format(num_models, fold))
