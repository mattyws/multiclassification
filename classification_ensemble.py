import csv
import json
import os
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
# TODO: check sync from dataset lists

def train_level_zero_classifiers(data, classes, model_creator, method="bagging"):
    if method == "bagging":
        #### START BAGGING ####
        ensemble = TrainEnsembleBagging(data, classes,
                                        model_creator=model_creator)
        ensemble.fit(epochs=parameters['level_0_epochs'])
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

if not os.path.exists(parameters['modelCheckpointPath']):
    os.mkdir(parameters['modelCheckpointPath'])

config = None
if os.path.exists(parameters['modelConfigPath']):
    with open(parameters['modelConfigPath'], 'r') as configHandler:
        config = json.load(configHandler)

# Loading csv
print_with_time("Loading data")
data_csv = pd.read_csv(parameters['dataset_csv_file_path'])
data_csv = data_csv.sort_values(['icustay_id'])

# If script is using structured data, do the preparation for it (normalization and get input shape)
structured_data = None
normalization_values = None
if parameters['use_structured_data']:
    print_with_time("Preparing structured data")
    structured_data = np.array([itemid for itemid in list(data_csv['icustay_id'])
                            if os.path.exists(parameters['structured_data_path'] + '{}.csv'.format(itemid))])
    print_with_time("Preparing normalization values")
    normalization_values = NormalizationValues(structured_data,
                                               pickle_object_path=parameters['normalization_value_counts_path'])
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
if parameters['use_textual_data']:
    print_with_time("Preparing textual data")
    textual_data = np.array([itemid for itemid in list(data_csv['icustay_id'])
                            if os.path.exists(parameters['textual_data_path'] + '{}.csv'.format(itemid))])
    word2vec_data = np.array([parameters['notes_word2vec_path'] + '{}.txt'.format(itemid) for itemid in textual_data])
    embedding_size = parameters['embedding_size']
    min_count = parameters['min_count']
    workers = parameters['workers']
    window = parameters['window']
    iterations = parameters['iterations']
    textual_input_shape = (None, None, embedding_size)

    print_with_time("Training/Loading Word2vec")
    preprocessing_pipeline = [escape_invalid_xml_characters, escape_html_special_entities, text_to_lower,
                              whitespace_tokenize_text, remove_only_special_characters_tokens, remove_sepsis_mentions]
    word2vec_model = train_representation_model(word2vec_data,
                                                parameters['word2vec_model_file_name'], min_count,
                                                embedding_size, workers, window, iterations)
    print_with_time("Transforming/Retrieving representation")
    texts_transformer = TransformClinicalTextsRepresentations(word2vec_model, embedding_size=embedding_size,
                                                              window=window, texts_path=parameters['dataPath'],
                                                              representation_save_path=parameters['textual_representation_data_path'],
                                                              text_max_len=228+224 # Valores com base na média + desvio padrão do tamanho dos textos já pre processados
                                                              )
    word2vec_model = None
    texts_transformer.transform(textual_data, preprocessing_pipeline=preprocessing_pipeline)
    textual_transformed_data = np.array(texts_transformer.get_new_paths(textual_data))
    # IN CASE THAT YOU ALREADY HAVE THE REPRESENTATIONS CREATED
    print_with_time("Padding/Retrieving sequences")
    # Valores com base na média + desvio padrão do tamanho dos textos já pre processados
    texts_transformer.pad_new_representation(textual_transformed_data, 228 + 224,
                                             pad_data_path=parameters['textual_padded_representation_data_path'])
    textual_transformed_data = np.array(texts_transformer.get_new_paths(textual_transformed_data))


# Using a seed always will get the same data split even if the training stops
print_with_time("Transforming classes")
classes = np.array([1 if c == 'sepsis' else 0 for c in list(data_csv['class'])])
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=15)
fold = 0
# ====================== Script that start training new models
with open(parameters['resultFilePath'], 'a+') as cvsFileHandler, \
        open(parameters['level_zero_result_file_path']) as level_zero_csv_file_handler: # where the results for each fold are appended
    dictWriter = None
    level_zero_dict_writer = None
    for trainIndex, testIndex in kf.split(structured_data, classes):
        if config is not None and config['fold'] > fold:
            print("Pass fold {}".format(fold))
            fold += 1
            continue
        print_with_time("Fold {}".format(fold))

        structured_ensemble = None
        if parameters['use_structured_data']:
            print_with_time("Getting values for normalization")
            values = normalization_values.get_normalization_values(structured_data[trainIndex],
                                                                   saved_file_name=parameters['normalized_data_path'].format(fold))
            normalizer = Normalization(values, temporary_path=parameters['normalized_structured_data_path'].format(fold))
            print_with_time("Normalizing fold data")
            normalizer.normalize_files(structured_data)
            normalized_data = np.array(normalizer.get_new_paths(structured_data))
            if not parameters['tcn']:
                modelCreator = MultilayerKerasRecurrentNNCreator(structured_input_shape, parameters['outputUnits'],
                                                                 parameters['numOutputNeurons'],
                                                                 loss=parameters['loss'],
                                                                 layersActivations=parameters['layersActivations'],
                                                                 networkActivation=parameters['networkActivation'],
                                                                 gru=parameters['gru'],
                                                                 use_dropout=parameters['useDropout'],
                                                                dropout=parameters['dropout'], kernel_regularizer=None,
                                                                 metrics=[keras.metrics.binary_accuracy],
                                                                 optimizer=parameters['optimizer'])
            else:
                modelCreator = MultilayerTemporalConvolutionalNNCreator(structured_input_shape, parameters['outputUnits'],
                                                                        parameters['numOutputNeurons'],
                                                                        loss=parameters['loss'],
                                                                        layersActivations=parameters[
                                                                            'layersActivations'],
                                                                        networkActivation=parameters[
                                                                            'networkActivation'],
                                                                        pooling=parameters['pooling'],
                                                                        kernel_sizes=parameters['kernel_sizes'],
                                                                        use_dropout=parameters['useDropout'],
                                                                        dilations=parameters['dilations'],
                                                                        nb_stacks=parameters['nb_stacks'],
                                                                        dropout=parameters['dropout'],
                                                                        kernel_regularizer=None,
                                                                        metrics=[keras.metrics.binary_accuracy],
                                                                        optimizer=parameters['optimizer'])
            print_with_time("Training level 0 models for structured data")
            structured_ensemble = train_level_zero_classifiers(normalized_data[trainIndex], classes[trainIndex],
                                                               modelCreator)

            test_sizes, test_labels = functions.divide_by_events_lenght(normalized_data[testIndex], classes[testIndex]
                                                                        , sizes_filename=parameters[
                    'testing_events_sizes_file'].format(fold)
                                                                        , classes_filename=parameters[
                    'testing_events_sizes_labels_file'].format(fold))
            dataTestGenerator = LengthLongitudinalDataGenerator(test_sizes, test_labels,
                                                                max_batch_size=parameters['batchSize'])
            dataTestGenerator.create_batches()
            print_with_time("Testing level 0 models for structured data")
            structured_level_zero_models = structured_ensemble.get_classifiers()
            for model in structured_level_zero_models:
                metrics = test_model(model, dataTestGenerator, fold)
                metrics['data_type'] = "structured"
                if level_zero_dict_writer is None:
                    level_zero_dict_writer = csv.DictWriter(level_zero_csv_file_handler, metrics.keys())
                if fold == 0:
                    level_zero_dict_writer.writeheader()
                level_zero_dict_writer.writerow(metrics)


        if parameters['use_textual_data']:
            modelCreator = NoteeventsClassificationModelCreator(textual_input_shape, parameters['outputUnits'],
                                                                parameters['numOutputNeurons'],
                                                                embedding_size=parameters['embedding_size'],
                                                                optimizer=parameters['optimizer'],
                                                                loss=parameters['loss'],
                                                                layersActivations=parameters['layersActivations'],
                                                                gru=parameters['gru'],
                                                                use_dropout=parameters['useDropout'],
                                                                dropout=parameters['dropout'],
                                                                networkActivation=parameters['networkActivation'],
                                                                metrics=[keras.metrics.binary_accuracy])
            print_with_time("Training level 0 models for textual data")
            textual_ensemble = train_level_zero_classifiers(textual_transformed_data[trainIndex], classes[trainIndex],
                                                               modelCreator)
            test_sizes, test_labels = functions.divide_by_events_lenght(textual_data[testIndex], classes[testIndex]
                                                                        , sizes_filename=parameters[
                    'testing_events_sizes_file'].format(fold)
                                                                        , classes_filename=parameters[
                    'testing_events_sizes_labels_file'].format(fold))
            dataTestGenerator = LengthLongitudinalDataGenerator(test_sizes, test_labels,
                                                                max_batch_size=parameters['batchSize'])
            dataTestGenerator.create_batches()
            print_with_time("Testing level 0 models for textual data")
            textual_level_zero_models = textual_ensemble.get_classifiers()
            for model in textual_level_zero_models:
                metrics = test_model(model, dataTestGenerator, fold)
                metrics['data_type'] = "textual"
                if level_zero_dict_writer is None:
                    level_zero_dict_writer = csv.DictWriter(level_zero_csv_file_handler, metrics.keys())
                if fold == 0:
                    level_zero_dict_writer.writeheader()
                level_zero_dict_writer.writerow(metrics)


        print_with_time("Preparing data to change their representation")
        if parameters['use_structured_data'] and parameters['use_textual_data']:
            meta_data = [ (parameters['normalized_data_path'] + itemid + '.csv',
                           parameters['textual_padded_representation_data_path'] + itemid + '.pkl')  for itemid in data_csv['icustay_id'] ]
            level_zero_models = []
            for model in structured_level_zero_models:
                level_zero_models.append((model, 0))
            for model in textual_level_zero_models:
                level_zero_models.append((model, 1))
        elif parameters['use_structured_data']:
            meta_data = normalized_data
            level_zero_models = structured_level_zero_models
        elif parameters['use_textual_data']:
            meta_data = textual_transformed_data
            level_zero_models = textual_level_zero_models

        print_with_time("Get model from adapters")
        aux_level_zero_models = []
        for adapter in level_zero_models:
            if isinstance(adapter, tuple):
                aux_level_zero_models.append((adapter[0].model, adapter[1]))
            else:
                aux_level_zero_models.append(adapter.model)


        print_with_time("Creating meta model data")

        meta_data_creator = EnsembleMetaLearnerDataCreator(level_zero_models)
        meta_data_creator.create_meta_learner_data(meta_data, parameters['meta_representation_path'])

        meta_data = meta_data_creator.get_new_paths(meta_data)


        print_with_time("Creating meta data generators")

        training_meta_data_generator = MetaLearnerDataGenerator(meta_data[trainIndex], classes[trainIndex],
                                                       batchSize=parameters['meta_learner_batch_size'])
        testing_meta_data_generator = MetaLearnerDataGenerator(meta_data[testIndex], classes[testIndex],
                                                                batchSize=parameters['meta_learner_batch_size'])

        meta_data_input_shape = (meta_data_creator.representation_length)
        modelCreator = EnsembleModelCreator(meta_data_input_shape, parameters['outputUnits'], parameters['numOutputNeurons'],
                                            loss=parameters['loss'], layers_activation=parameters['layersActivations'],
                                            network_activation=parameters['networkActivation'],
                                            use_dropout=parameters['useDropout'],
                                            dropout=parameters['dropout'],
                                            metrics=[keras.metrics.binary_accuracy], optimizer=parameters['optimizer'])
        with open(parameters['modelCheckpointPath']+"parameters.json", 'w') as handler:
            json.dump(parameters, handler)
        kerasAdapter = modelCreator.create()
        epochs = parameters['trainingEpochs']
        print_with_time("Training model")
        kerasAdapter.fit(training_meta_data_generator, epochs=epochs, callbacks=None)
        print_with_time("Testing model")
        metrics = test_model(kerasAdapter, testing_meta_data_generator, fold)
        if dictWriter is None:
            dictWriter = csv.DictWriter(cvsFileHandler, metrics.keys())
        if fold == 0:
            dictWriter.writeheader()
        dictWriter.writerow(metrics)
        kerasAdapter.save(parameters['modelCheckpointPath'] + 'trained_{}.model'.format(fold))
        fold += 1
