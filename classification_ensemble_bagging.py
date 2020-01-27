import csv
import datetime
import json
import os
import pickle
from collections import Counter
from pprint import PrettyPrinter

import pandas as pd
import numpy as np

import keras
from keras import backend as K
import tensorflow as tf
from keras.regularizers import l1_l2

from sklearn.model_selection._split import StratifiedKFold

import functions
from adapter import KerasAdapter
from data_generators import LengthLongitudinalDataGenerator, LongitudinalDataGenerator, MetaLearnerDataGenerator
from data_representation import EnsembleMetaLearnerDataCreator, TransformClinicalTextsRepresentations
from ensemble_training import TrainEnsembleAdaBoosting, TrainEnsembleBagging
from functions import test_model, print_with_time, escape_invalid_xml_characters, escape_html_special_entities, \
    text_to_lower, remove_sepsis_mentions, remove_only_special_characters_tokens, whitespace_tokenize_text, \
    train_representation_model
from keras_callbacks import Metrics
from model_creators import MultilayerKerasRecurrentNNCreator, EnsembleModelCreator, \
    MultilayerTemporalConvolutionalNNCreator, NoteeventsClassificationModelCreator
from normalization import Normalization, NormalizationValues

# TODO: check resuming weak model training
# TODO: check saving path

def train_level_zero_classifiers(data, classes, model_creator, training_data_samples=None, training_classes_samples=None,
                                 level_zero_epochs=20, n_estimators=10, batch_size=30, method="bagging", split_rate=.4,
                                 saved_model_path="level_zero_model_{}.model", data_samples_path="bagging_samples_{}.model"):
    if method == "bagging":
        #### START BAGGING ####
        ensemble = TrainEnsembleBagging()
        ensemble.fit(data, classes, model_creator, training_data_samples=training_data_samples, split_rate=split_rate,
                     training_classes_samples=training_classes_samples, epochs=level_zero_epochs,
                     batch_size=batch_size, n_estimators=n_estimators, saved_model_path=saved_model_path,
                     saved_data_samples_path=data_samples_path)
        ### END ADABOOSTING ####
    elif method == "clustering":
        ### START CLUSTERING ENSEMBLE ###
        ensemble = None
        ### END CLUSTERING ENSEMBLE ###
    else:
        raise ValueError("Either bagginng or clustering")
    return ensemble

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
DATETIME_PATTERN = "%Y-%m-%d %H:%M:%S"
from classification_ensemble_parameters import parameters


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
    word2vec_data = np.array([parameters['notes_word2vec_path'] + '{}.txt'.format(itemid) for itemid in textual_data])
    embedding_size = parameters['textual_embedding_size']
    min_count = parameters['textual_min_count']
    workers = parameters['textual_workers']
    window = parameters['textual_window']
    iterations = parameters['textual_iterations']
    textual_input_shape = (None, None, embedding_size)

    print_with_time("Training/Loading Word2vec")
    preprocessing_pipeline = [escape_invalid_xml_characters, escape_html_special_entities, text_to_lower,
                              whitespace_tokenize_text, remove_only_special_characters_tokens, remove_sepsis_mentions]
    word2vec_model = train_representation_model(word2vec_data,
                                                parameters['training_directory_path'] +
                                                parameters['word2vec_model_file_name'],
                                                min_count,
                                                embedding_size, workers, window, iterations)
    print_with_time("Transforming/Retrieving representation")
    texts_transformer = TransformClinicalTextsRepresentations(word2vec_model, embedding_size=embedding_size,
                                                              window=window, texts_path=parameters['textual_data_path'],
                                                              representation_save_path=parameters['training_directory_path']
                                                                                       + parameters['word2vec_representation_data_path']
                                                              )
    word2vec_model = None
    texts_transformer.transform(textual_data, preprocessing_pipeline=preprocessing_pipeline)
    textual_transformed_data = np.array(texts_transformer.get_new_paths(textual_data))
    # IN CASE THAT YOU ALREADY HAVE THE REPRESENTATIONS CREATED
    print_with_time("Padding/Retrieving sequences")
    # Valores com base na média + desvio padrão do tamanho dos textos já pre processados
    texts_transformer.pad_new_representation(textual_transformed_data, 228 + 224,
                                             pad_data_path=parameters['training_directory_path'] +
                                                           parameters['word2vec_padded_representation_files_path'])
    textual_transformed_data = np.array(texts_transformer.get_new_paths(textual_transformed_data))


# Using a seed always will get the same data split even if the training stops
print_with_time("Transforming classes")
classes = np.array([1 if c == 'sepsis' else 0 for c in list(data_csv['class'])])
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=15)
fold = 0
# ====================== Script that start training new models
with open(parameters['training_directory_path'] + parameters['checkpoint'] + parameters['results_file_name'], 'a+') as cvsFileHandler, \
        open(parameters['training_directory_path'] + parameters['checkpoint'] + parameters['level_zero_result_file_name'], 'a+')\
                as level_zero_csv_file_handler: # where the results for each fold are appended
    dictWriter = None
    level_zero_dict_writer = None
    for trainIndex, testIndex in kf.split(structured_data, classes):
        jump_fold = True
        for num_models in range(1, parameters['n_estimators']):
            if not os.path.exists(parameters['training_directory_path'] + parameters['checkpoint']
                              + parameters['meta_model_file_name'].format(num_models, fold)):
                jump_fold = False
        if jump_fold:
            print("Pass fold {}".format(fold))
            fold += 1
            continue
        print_with_time("Fold {}".format(fold))

        structured_ensemble = None
        if parameters['use_structured_data']:
            print_with_time("Getting values for normalization")
            values = normalization_values.get_normalization_values(structured_data[trainIndex],
                                                                   saved_file_name=parameters['training_directory_path']
                                                                                   + parameters['normalization_data_path'].format(fold))
            normalizer = Normalization(values, temporary_path=parameters['training_directory_path']
                                                              + parameters['normalized_structured_data_path'].format(fold))
            print_with_time("Normalizing fold data")
            normalizer.normalize_files(structured_data)
            normalized_data = np.array(normalizer.get_new_paths(structured_data))
            if not parameters['structured_tcn']:
                modelCreator = MultilayerKerasRecurrentNNCreator(structured_input_shape, parameters['structured_output_units'],
                                                                 parameters['structured_output_neurons'],
                                                                 loss=parameters['structured_loss'],
                                                                 layersActivations=parameters['structured_layers_activations'],
                                                                 networkActivation=parameters['structured_network_activation'],
                                                                 gru=parameters['structured_gru'],
                                                                 use_dropout=parameters['structured_use_dropout'],
                                                                dropout=parameters['structured_dropout'], kernel_regularizer=None,
                                                                 metrics=[keras.metrics.binary_accuracy],
                                                                 optimizer=parameters['structured_optimizer'])
            else:
                modelCreator = MultilayerTemporalConvolutionalNNCreator(structured_input_shape, parameters['structured_output_units'],
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
                                                                        kernel_regularizer=None,
                                                                        metrics=[keras.metrics.binary_accuracy],
                                                                        optimizer=parameters['structured_optimizer'])
            print_with_time("Training level 0 models for structured data")
            level_zero_models_saving_path = parameters['training_directory_path'] + parameters['ensemble_models_path'].format(fold)
            if not os.path.exists(level_zero_models_saving_path):
                os.mkdir(level_zero_models_saving_path)
            start = datetime.datetime.now()
            structured_ensemble = train_level_zero_classifiers(normalized_data[trainIndex], classes[trainIndex],
                                                               modelCreator, level_zero_epochs=parameters['structured_training_epochs'],
                                                               n_estimators=parameters['n_estimators'],
                                                               split_rate=parameters['dataset_split_rate'],
                                                               batch_size=parameters['structured_batch_size'],
                                                               saved_model_path=level_zero_models_saving_path +
                                                                                parameters['structured_ensemble_models_name_prefix'],
                                                               data_samples_path = level_zero_models_saving_path +
                                                                                parameters['structured_ensemble_samples_name_prefix'])
            end = datetime.datetime.now()
            time_to_train = end - start
            hours, remainder = divmod(time_to_train.seconds, 3600)
            minutes, seconds = divmod(remainder, 60)
            print_with_time('Took {:02}:{:02}:{:02} to train the level zero models for structured data'.format(int(hours), int(minutes), int(seconds)))

            if not os.path.exists(parameters['training_directory_path'] + parameters['checkpoint'] +
                                      parameters['level_zero_structured_result_file_name'].format(fold)):
                print_with_time("Testing level 0 models for structured data")
                test_sizes, test_labels = functions.divide_by_events_lenght(normalized_data[testIndex],
                                                                            classes[testIndex],
                                                                            sizes_filename=parameters['training_directory_path'] +
                                                                                           parameters['structured_testing_events_sizes_file'].format(fold)
                                                                            , classes_filename=parameters['training_directory_path'] +
                                                                                               parameters['structured_testing_events_sizes_labels_file'].format(fold))
                dataTestGenerator = LengthLongitudinalDataGenerator(test_sizes, test_labels,
                                                                    max_batch_size=parameters['structured_batch_size'])
                dataTestGenerator.create_batches()
                structured_level_zero_models = structured_ensemble.get_classifiers()
                structured_results = []
                for i, model in enumerate(structured_level_zero_models):
                    metrics = test_model(model, dataTestGenerator, fold)
                    metrics['model_num'] = i
                    structured_results.append(metrics)
                structured_results = pd.DataFrame(structured_results)
                print(structured_results)
                structured_results.to_csv(parameters['training_directory_path'] + parameters['checkpoint'] +
                                          parameters['level_zero_structured_result_file_name'].format(fold))


        if parameters['use_textual_data']:
            modelCreator = NoteeventsClassificationModelCreator(textual_input_shape, parameters['textual_output_units'],
                                                                parameters['textual_output_neurons'],
                                                                embedding_size=parameters['textual_embedding_size'],
                                                                optimizer=parameters['textual_optimizer'],
                                                                loss=parameters['textual_loss'],
                                                                layersActivations=parameters['textual_layers_activations'],
                                                                use_dropout=parameters['textual_use_dropout'],
                                                                dropout=parameters['textual_dropout'],
                                                                networkActivation=parameters['textual_network_activation'],
                                                                metrics=[keras.metrics.binary_accuracy])
            print_with_time("Training level 0 models for textual data")
            training_data_samples = None
            training_classes_samples = None,
            if parameters['use_structured_data']:
                training_data_samples = structured_ensemble.training_data_samples
                training_data_samples = [ [path.replace(parameters['normalized_structured_data_path'].format(fold),
                                parameters['word2vec_padded_representation_files_path']) for path in samples ]
                  for samples in training_data_samples]
                training_classes_samples = structured_ensemble.training_classes_samples
            level_zero_models_saving_path = parameters['training_directory_path'] \
                                            + parameters['ensenble_models_path'].format(fold)
            if not os.path.exists(level_zero_models_saving_path):
                os.mkdir(level_zero_models_saving_path)
            start = datetime.datetime.now()
            textual_ensemble = train_level_zero_classifiers(normalized_data[trainIndex], classes[trainIndex],
                                                               modelCreator, level_zero_epochs=parameters['textual_training_epochs'],
                                                               n_estimators=parameters['n_estimators'],
                                                               batch_size=parameters['textual_batch_size'],
                                                               split_rate=parameters['dataset_split_rate'],
                                                               saved_model_path=level_zero_models_saving_path
                                                                                + parameters['textual_ensemble_models_name_prefix'],
                                                            data_samples_path=level_zero_models_saving_path +
                                                                              parameters['structured_ensemble_samples_name_prefix'],
                                                            training_data_samples=training_data_samples,
                                                            training_classes_samples=training_classes_samples)
            end = datetime.datetime.now()
            time_to_train = end - start
            hours, remainder = divmod(time_to_train.seconds, 3600)
            minutes, seconds = divmod(remainder, 60)
            print_with_time(
                'Took {:02}:{:02}:{:02} to train the level zero models for textual data'.format(int(hours),
                                                                                                   int(minutes),
                                                                                                   int(seconds)))
            if not os.path.exists(parameters['training_directory_path'] + parameters['checkpoint'] +
                                      parameters['level_zero_textual_result_file_name'].format(fold)):
                print_with_time("Testing level 0 models for textual data")
                test_sizes, test_labels = functions.divide_by_events_lenght(textual_data[testIndex], classes[testIndex]
                                                                            , sizes_filename=parameters['training_directory_path'] +
                                                                                             parameters['textual_testing_events_sizes_file'].format(fold)
                                                                            , classes_filename=parameters['training_directory_path'] +
                                                                                                parameters['textual_testing_events_sizes_labels_file'].format(fold))
                dataTestGenerator = LengthLongitudinalDataGenerator(test_sizes, test_labels,
                                                                    max_batch_size=parameters['textual_batch_size'])
                dataTestGenerator.create_batches()
                textual_results = []
                textual_level_zero_models = textual_ensemble.get_classifiers()
                for i, model in enumerate(textual_level_zero_models):
                    metrics = test_model(model, dataTestGenerator, fold)
                    metrics['model_num'] = i
                    textual_results.append(metrics)
                textual_results = pd.DataFrame(structured_results)
                print(textual_results)
                textual_results.to_csv(parameters['training_directory_path'] + parameters['checkpoint'] +
                                          parameters['level_zero_textual_result_file_name'].format(fold))


        for num_models in range(1, parameters['n_estimators']):
            print_with_time("Preparing data to change their representation")
            if parameters['use_structured_data'] and parameters['use_textual_data']:
                meta_data = [ (parameters['training_directory_path'] + parameters['normalized_data_path'] + itemid + '.csv',
                               parameters['training_directory_path'] + parameters['textual_padded_representation_data_path'] + itemid + '.pkl')
                              for itemid in data_csv['icustay_id'] ]
                level_zero_models = []
                for model_num, model in enumerate(structured_level_zero_models):
                    level_zero_models.append((model, 0))
                    if model_num == num_models:
                        break
                for model_num, model in enumerate(textual_level_zero_models):
                    level_zero_models.append((model, 1))
                    if model_num == num_models:
                        break
            elif parameters['use_structured_data']:
                meta_data = normalized_data
                level_zero_models = structured_level_zero_models[: num_models]
            elif parameters['use_textual_data']:
                meta_data = textual_transformed_data
                level_zero_models = textual_level_zero_models[: num_models]

            print_with_time("Get model from adapters")
            aux_level_zero_models = []
            for adapter in level_zero_models:
                if isinstance(adapter, tuple):
                    aux_level_zero_models.append((adapter[0].model, adapter[1]))
                else:
                    aux_level_zero_models.append(adapter.model)
            level_zero_models = aux_level_zero_models

            print_with_time("Creating meta model data")

            meta_data_creator = EnsembleMetaLearnerDataCreator(level_zero_models)
            meta_data_creator.create_meta_learner_data(meta_data, parameters['training_directory_path'] +
                                                       parameters['meta_representation_path'].format(num_models, fold))

            meta_data = meta_data_creator.get_new_paths(meta_data)

            print_with_time("Creating meta data generators")

            training_meta_data_generator = MetaLearnerDataGenerator(meta_data[trainIndex], classes[trainIndex],
                                                           batchSize=parameters['meta_learner_batch_size'])
            testing_meta_data_generator = MetaLearnerDataGenerator(meta_data[testIndex], classes[testIndex],
                                                                    batchSize=parameters['meta_learner_batch_size'])

            meta_data_input_shape = (parameters['meta_learner_batch_size'], meta_data_creator.representation_length)
            modelCreator = EnsembleModelCreator(meta_data_input_shape, parameters['outputUnits'], parameters['numOutputNeurons'],
                                                loss=parameters['loss'], layers_activation=parameters['layersActivations'],
                                                network_activation=parameters['networkActivation'],
                                                use_dropout=parameters['useDropout'],
                                                dropout=parameters['dropout'],
                                                metrics=[keras.metrics.binary_accuracy], optimizer=parameters['optimizer'])
            kerasAdapter = modelCreator.create()
            epochs = parameters['trainingEpochs']
            print_with_time("Training model")
            start = datetime.datetime.now()
            kerasAdapter.fit(training_meta_data_generator, epochs=epochs, callbacks=None)
            end = datetime.datetime.now()
            time_to_train = end - start
            hours, remainder = divmod(time_to_train.seconds, 3600)
            minutes, seconds = divmod(remainder, 60)
            print_with_time('Took {:02}:{:02}:{:02} to train the model'.format(int(hours), int(minutes), int(seconds)))
            print_with_time("Testing model")
            metrics = test_model(kerasAdapter, testing_meta_data_generator, fold)
            metrics['num_models'] = num_models
            if dictWriter is None:
                dictWriter = csv.DictWriter(cvsFileHandler, metrics.keys())
            if fold == 0:
                dictWriter.writeheader()
            dictWriter.writerow(metrics)
            kerasAdapter.save(parameters['training_directory_path'] + parameters['checkpoint']
                              + parameters['meta_model_file_name'].format(num_models, fold))
            fold += 1
