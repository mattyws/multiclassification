import csv
import datetime
import json
import os
import pickle

import pandas
import pandas as pd
import numpy as np
import tensorflow as tf

import functions
from adapter import KerasAdapter
from data_generators import LengthLongitudinalDataGenerator, LongitudinalDataGenerator, MetaLearnerDataGenerator, \
    ArrayDataGenerator, MixedLengthDataGenerator
from data_representation import EnsembleMetaLearnerDataCreator, TransformClinicalTextsRepresentations
from ensemble_training import TrainEnsembleAdaBoosting, TrainEnsembleBagging, split_classes
from functions import test_model, print_with_time, escape_invalid_xml_characters, escape_html_special_entities, \
    text_to_lower, remove_sepsis_mentions, remove_only_special_characters_tokens, whitespace_tokenize_text, \
    train_representation_model, remove_columns_for_classification
from keras_callbacks import Metrics
from model_creators import MultilayerKerasRecurrentNNCreator, EnsembleModelCreator, \
    MultilayerTemporalConvolutionalNNCreator, NoteeventsClassificationModelCreator, KerasTunerModelCreator, \
    MixedInputModelCreator, MultilayerTemporalConvolutionalNNHyperModel, MultilayerKerasRecurrentNNHyperModel
from normalization import Normalization, NormalizationValues
import kerastuner as kt
import tensorflow.keras as keras
import sys
from keras.regularizers import l1_l2
from sklearn.linear_model.logistic import LogisticRegression

from sklearn.model_selection._split import StratifiedKFold, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network.multilayer_perceptron import MLPClassifier
from sklearn.svm.classes import LinearSVC
from sklearn.tree.tree import DecisionTreeClassifier
from sklearn.utils import class_weight
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l1
from tensorflow.keras.metrics import AUC

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from result_evaluation import ModelEvaluation

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
DATETIME_PATTERN = "%Y-%m-%d %H:%M:%S"
from multiclassification.parameters.classification_parameters import ensemble_stacking_parameters as parameters

problem = 'mortality'
training_base_directory = parameters['multiclassification_base_path'] + parameters['training_directory_path']
training_directory = training_base_directory + parameters[problem+"_directory"] \
                     + parameters['execution_saving_path']
checkpoint_directory = training_directory + parameters['training_checkpoint']

if not os.path.exists(checkpoint_directory):
    os.makedirs(checkpoint_directory)

with open(checkpoint_directory + "parameters.pkl", 'wb') as handler:
    pickle.dump(parameters, handler)

# Loading csv
print_with_time("Loading data")
dataset_path = parameters['multiclassification_base_path'] + parameters[problem+'_directory'] \
               + parameters[problem+'_dataset_csv']

data_csv = pd.read_csv(dataset_path)
data_csv = data_csv.sort_values(['episode'])

print_with_time("Preparing structured data")
structured_data = np.asarray(data_csv['structured_path'].tolist())
print_with_time("Preparing normalization values")
normalization_value_counts_path = training_directory + parameters['normalization_value_counts_dir']
normalization_values = NormalizationValues(structured_data,
                                           pickle_object_path= normalization_value_counts_path)
normalization_values.prepare()
# Get input shape
aux = pd.read_csv(structured_data[0])
aux = remove_columns_for_classification(aux)
structured_input_shape = (None, len(aux.columns))

print_with_time("Preparing textual data")
textual_data = np.array(data_csv['textual_path'].tolist())
print("Len textual data", len(textual_data), len(np.array(data_csv['textual_path'].dropna().tolist())))
# word2vec_data = np.array([parameters['notes_word2vec_path'] + '{}.txt'.format(itemid) for itemid in textual_data])
embedding_size = parameters['textual_embedding_size']
min_count = parameters['textual_min_count']
workers = parameters['textual_workers']
window = parameters['textual_window']
iterations = parameters['textual_iterations']
textual_input_shape = (None, embedding_size)

print_with_time("Training/Loading representation model")
preprocessing_pipeline = [whitespace_tokenize_text]
texts_hourly_merged_dir = parameters['multiclassification_base_path'] + "textual_hourly_merged/"
representation_model_data = [texts_hourly_merged_dir + x for x in os.listdir(texts_hourly_merged_dir)]
textual_representation_model_path = parameters['textual_representation_model_path'] + \
                                     "{}/{}".format(embedding_size, parameters['textual_representation_model_filename'])
if not os.path.exists(parameters['textual_representation_model_path'] + "{}/".format(embedding_size)):
    os.makedirs(parameters['textual_representation_model_path'] + "{}/".format(embedding_size))
representation_model = train_representation_model(representation_model_data,
                                                textual_representation_model_path,
                                                  min_count, embedding_size, workers, window, iterations,
                                                  hs=parameters['textual_doc2vec_hs'], dm=parameters['textual_doc2vec_dm'],
                                                  negative=parameters['textual_doc2vec_negative'],
                                                  preprocessing_pipeline=preprocessing_pipeline, word2vec=False)
print_with_time("Transforming/Retrieving representation")
notes_textual_representation_path = parameters['textual_representation_model_path'] \
                                    + "{}/{}/{}".format(embedding_size, problem, parameters['notes_textual_representation_directory'])
if not os.path.exists(notes_textual_representation_path):
    os.makedirs(notes_textual_representation_path)
texts_transformer = TransformClinicalTextsRepresentations(representation_model, embedding_size=embedding_size,
                                                          window=window,
                                                          representation_save_path=notes_textual_representation_path,
                                                          is_word2vec=False)
representation_model = None
texts_transformer.transform(textual_data, preprocessing_pipeline=preprocessing_pipeline)
# textual_transformed_data = np.asarray(texts_transformer.get_new_paths(textual_data))

# Using a seed always will get the same data split even if the training stops
print_with_time("Transforming classes")
classes = np.asarray(data_csv['label'].tolist())

print_with_time("Training/evaluation data spliting")
data_csv.loc[:, 'episode'] = data_csv['episode'].astype(str)
episodes = np.asarray(data_csv['episode'].tolist())

training_samples_path = training_directory + parameters['training_samples_filename']
training_classes_path = training_directory + parameters['training_classes_filename']
testing_samples_path = training_directory + parameters['testing_samples_filename']
testing_classes_path = training_directory + parameters['testing_classes_filename']

structured_model_creator = None
textual_model_creator = None

# parameters['model_tunning'] = False
if parameters['model_tunning']:
    optimization_samples_path = training_directory + parameters['optimization_samples_filename']
    optimization_classes_path = training_directory + parameters['optimization_classes_filename']
    from multiclassification.parameters.classification_parameters import timeseries_tuning_parameters, textual_tuning_parameters
    optimization_smodel_hypermodels = MultilayerTemporalConvolutionalNNHyperModel(structured_input_shape,
                                                               parameters['structured_output_neurons'],
                                                                [AUC(name='auc')], timeseries_tuning_parameters,
                                                              model_input_name= parameters['structured_model_input_name'])
    optimization_tmodel_hypermodels = MultilayerTemporalConvolutionalNNHyperModel(textual_input_shape,
                                                                                       parameters['textual_output_neurons'],
                                                                                       [AUC(name='auc')],
                                                                                       textual_tuning_parameters,
                                                              model_input_name=parameters['textual_model_input_name'])
    if not os.path.exists(training_samples_path):
        len_dataset = len(episodes)
        len_optimization_dataset = int(len_dataset * parameters['optimization_split_rate'])
        episodes, episodes_opt, classes, classes_opt = train_test_split(episodes, classes, stratify=classes,
                                                                 test_size=len_optimization_dataset)
        with open(training_samples_path, 'wb') as f:
            pickle.dump(episodes, f)
        with open(training_classes_path, 'wb') as f:
            pickle.dump(classes, f)
        with open(optimization_samples_path, 'wb') as f :
            pickle.dump(episodes_opt, f)
        with open(optimization_classes_path, 'wb') as f:
            pickle.dump(classes_opt, f)
    else:
        with open(training_samples_path, 'rb') as f:
            episodes = pickle.load(f)
        with open(training_classes_path, 'rb') as f:
            classes = pickle.load(f)
        with open(optimization_samples_path, 'rb') as f:
            episodes_opt = pickle.load(f)
        with open(optimization_classes_path, 'rb') as f:
            classes_opt = pickle.load(f)

    optimization_df = data_csv[data_csv['episode'].isin(episodes_opt)]
    classes_opt = np.asarray(optimization_df['label'].tolist())
    optimization_sdata = np.asarray(optimization_df['structured_path'].tolist())
    optimization_tdata = np.asarray(optimization_df['textual_path'].tolist())

    print_with_time("Get transformed textual data")
    opt_textual_transformed_data = np.asarray(texts_transformer.get_new_paths(optimization_tdata))

    opt_normalization_values_path = training_directory + parameters['optimization_normalization_values_filename']
    values = normalization_values.get_normalization_values(optimization_sdata,
                                                           saved_file_name=opt_normalization_values_path)
    opt_normalization_temporary_data_path = training_directory + parameters[
        'optimization_normalization_temporary_data_directory']
    opt_normalizer = Normalization(values, temporary_path=opt_normalization_temporary_data_path)
    print_with_time("Normalizing optimization data")
    opt_normalizer.normalize_files(optimization_sdata)
    opt_normalized_data = np.array(opt_normalizer.get_new_paths(optimization_sdata))

    print_with_time("Creating structured optimization generators")
    training_events_sizes_file_path = training_directory + parameters['structured_training_events_sizes_filename'].format('sopt')
    training_events_sizes_labels_file_path = training_directory + parameters[
        'structured_training_events_sizes_labels_filename'].format('sopt')

    train_sizes, train_labels = functions.divide_by_events_lenght(opt_normalized_data
                                                                  , classes_opt
                                                                  , sizes_filename=training_events_sizes_file_path
                                                                  ,
                                                                  classes_filename=training_events_sizes_labels_file_path)
    dataTrainGenerator = LengthLongitudinalDataGenerator(train_sizes, train_labels,
                                                         max_batch_size=parameters['structured_batch_size'])
    dataTrainGenerator.create_batches()

    tunning_directory = checkpoint_directory + parameters['tunning_directory']
    tuner = kt.Hyperband(optimization_smodel_hypermodels,
                         objective=kt.Objective('auc', direction="max"),
                         max_epochs=10,
                         directory=tunning_directory,
                         project_name=optimization_smodel_hypermodels.name,
                         factor=3)
    tuner.search(dataTrainGenerator, epochs=10)
    structured_model_creator = KerasTunerModelCreator(tuner,
                                                      optimization_smodel_hypermodels,
                                                      optimization_smodel_hypermodels.name,
                                                      remove_last_layer=False)

    training_events_sizes_file_path = training_directory \
                                      + parameters['textual_training_events_sizes_filename'].format('topt')
    training_events_sizes_labels_file_path = training_directory \
                                             + parameters['textual_training_events_sizes_labels_filename'].format('topt')

    train_sizes, train_labels = functions.divide_by_events_lenght(opt_textual_transformed_data
                                                                  , classes_opt
                                                                  , sizes_filename=training_events_sizes_file_path
                                                                  ,
                                                                  classes_filename=training_events_sizes_labels_file_path)

    dataTrainGenerator = LengthLongitudinalDataGenerator(train_sizes, train_labels,
                                                         max_batch_size=parameters['textual_batch_size'])
    dataTrainGenerator.create_batches()
    optimization_tmodel_hypermodels.name = optimization_tmodel_hypermodels.name + '_textual'
    tunning_directory = checkpoint_directory + parameters['tunning_directory']
    tuner = kt.Hyperband(optimization_tmodel_hypermodels,
                         objective=kt.Objective('auc', direction="max"),
                         max_epochs=10,
                         directory=tunning_directory,
                         project_name=optimization_tmodel_hypermodels.name,
                         factor=3)
    tuner.search(dataTrainGenerator, epochs=10)
    textual_model_creator = KerasTunerModelCreator(tuner,
                                                   optimization_tmodel_hypermodels,
                                                   optimization_tmodel_hypermodels.name,
                                                   remove_last_layer=False)
else :
    structured_model_creator = MultilayerTemporalConvolutionalNNCreator(structured_input_shape,
                                                            parameters['structured_output_units'],
                                                            None,
                                                            loss=parameters['structured_loss'],
                                                            layersActivations=parameters[
                                                                'structured_layers_activations'],
                                                            networkActivation=parameters[
                                                                'structured_network_activation'],
                                                            pooling=parameters['structured_pooling'],
                                                            kernel_sizes=parameters['structured_kernel_sizes'],
                                                            use_dropout=False,
                                                            dilations=parameters['structured_dilations'],
                                                            nb_stacks=parameters['structured_nb_stacks'],
                                                            dropout=parameters['structured_dropout'],
                                                            kernel_regularizer=None,
                                                            metrics=[keras.metrics.binary_accuracy, AUC()],
                                                            model_input_name=parameters['structured_model_input_name'],
                                                            optimizer=parameters['structured_optimizer'])
    structured_model_creator.create(model_summary_filename="structured_summary.txt")
    textual_model_creator = MultilayerTemporalConvolutionalNNCreator(textual_input_shape, parameters['textual_output_units'],
                                                            None,
                                                            loss=parameters['textual_loss'],
                                                            layersActivations=parameters['textual_layers_activations'],
                                                            networkActivation=parameters['textual_network_activation'],
                                                            pooling=parameters['textual_pooling'],
                                                            dilations=parameters['textual_dilations'],
                                                            nb_stacks=parameters['textual_nb_stacks'],
                                                            kernel_sizes=parameters['textual_kernel_sizes'],
                                                            kernel_regularizer=l1(0.01),
                                                            use_dropout=False,
                                                            dropout=parameters['textual_dropout'],
                                                            metrics=[keras.metrics.binary_accuracy, AUC()],
                                                         model_input_name=parameters['textual_model_input_name'],
                                                            optimizer=parameters['textual_optimizer'])
    textual_model_creator.create(model_summary_filename="textual_summary.txt")

training_df = data_csv[data_csv['episode'].isin(episodes)]
classes = np.asarray(training_df['label'].tolist())
episodes = np.asarray(training_df['episode'].tolist())

structured_data = np.asarray(training_df['structured_path'].tolist())
textual_data = np.asarray(training_df['textual_path'].tolist())

textual_transformed_data = np.asarray(texts_transformer.get_new_paths(textual_data))
training_df['textual_transformed_path'] = textual_transformed_data


kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=15)
fold = 0


all_metrics = pd.DataFrame({})
# ====================== Script that start training new models
results_file_path = checkpoint_directory + parameters['metrics_filename']
level_zero_results_file_path = checkpoint_directory + parameters['level_zero_result_filename']

dictWriter = None
level_zero_dict_writer = None
for trainIndex, testIndex in kf.split(episodes, classes):
    print(len(trainIndex), len(testIndex))
    print(len(structured_data[trainIndex]), len(structured_data[testIndex]))
    print(len(textual_transformed_data[trainIndex]), len(textual_transformed_data[testIndex]))
    print(len(classes[trainIndex]), len(classes[testIndex]))

    trained_model_path = checkpoint_directory + parameters['structured_weak_model_all'].format(fold)
    if os.path.exists(trained_model_path):
        print("Pass fold {}".format(fold))
        all_metrics = pandas.read_csv(results_file_path)
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
    class_weights = None

    print_with_time("Getting values for normalization")
    fold_normalization_values_path = training_directory + parameters['fold_normalization_values_filename'].format(fold)
    values = normalization_values.get_normalization_values(structured_data[trainIndex],
                                                           saved_file_name=fold_normalization_values_path)
    fold_normalization_temporary_data_path = training_directory \
                                             + parameters['fold_normalization_temporary_data_directory'].format(fold)
    normalizer = Normalization(values, temporary_path=fold_normalization_temporary_data_path)

    print_with_time("Normalizing fold data")
    normalizer.normalize_files(structured_data)
    normalized_data = np.array(normalizer.get_new_paths(structured_data))

    training_df['structured_normalized_path'] = normalized_data
    print(training_df[['episode', 'structured_normalized_path']])
    print(training_df[['episode', 'textual_transformed_path']])
    print(training_df[['episode', 'label']])
    training_folds_df = training_df[training_df['episode'].isin(episodes[trainIndex])]
    testing_folds_df = training_df[training_df['episode'].isin(episodes[testIndex])]

    training_events_sizes_file_path = training_directory \
                                      + parameters['structured_training_events_sizes_filename'].format(fold)
    testing_events_sizes_file_path = training_directory \
                                     + parameters['structured_testing_events_sizes_filename'].format(fold)

    train_sizes = functions.mixed_divide_by_events_lenght(training_folds_df, 'structured_normalized_path',
                                                          sizes_filename=training_events_sizes_file_path)
    test_sizes = functions.mixed_divide_by_events_lenght(testing_folds_df, 'structured_normalized_path',
                                                         sizes_filename=testing_events_sizes_file_path)
    dataTrainGenerator = MixedLengthDataGenerator(training_folds_df, max_batch_size=parameters['structured_batch_size'],
                                                  structured_df_column="structured_normalized_path",
                                                  textual_df_column="textual_transformed_path",
                                                  structured_model_input_name=parameters['structured_model_input_name'],
                                                  textual_model_input_name=parameters['textual_model_input_name'])
    dataTrainGenerator.create_batches(train_sizes)

    dataTestGenerator = MixedLengthDataGenerator(testing_folds_df, max_batch_size=parameters['structured_batch_size'],
                                                 structured_df_column="structured_normalized_path",
                                                 textual_df_column="textual_transformed_path",
                                                 structured_model_input_name=parameters['structured_model_input_name'],
                                                 textual_model_input_name=parameters['textual_model_input_name'])
    dataTestGenerator.create_batches(test_sizes)
    # print(test_sizes)
    # input()
    # for i in range(len(dataTestGenerator)):
    #     print(dataTestGenerator[i])
    # exit()

    start = datetime.datetime.now()
    # Getting models
    graph = tf.Graph()
    # with graph.as_default():
    structured_model = structured_model_creator.create().model
    textual_model = textual_model_creator.create().model

    mixed_model_creator = MixedInputModelCreator( [structured_model, textual_model],
                                                  parameters['meta_learner_output_units'],
                                                  parameters['meta_learner_num_output_neurons'],
                                                  layersActivations=parameters['meta_learner_layers_activations'],
                                                  networkActivation=parameters['meta_learner_network_activation'],
                                                  loss=parameters['meta_learner_loss'],
                                                  optimizer=parameters['meta_learner_optimizer'],
                                                  use_dropout=parameters['meta_learner_use_dropout'],
                                                  dropout=parameters['meta_learner_dropout'],
                                                  metrics=[AUC(name='auc')],
                                                  kernel_regularizer=None,
                                                  bias_regularizer=None,
                                                  activity_regularizer=None)

    if not os.path.exists(trained_model_path):
        kerasAdapter = mixed_model_creator.create(model_summary_filename=checkpoint_directory + 'model_summary.txt')
        epochs = parameters['meta_learner_training_epochs']
        print_with_time("Training model")
        # structured_data = []
        # textual_data = []
        # for sample_num in range(10):
        #     structured_sample = np.random.randint(4, size=(1280, 72)).astype('float32')
        #     textual_sample = np.random.randint(4, size=(1280, embedding_size)).astype('float32')
        #     structured_data.append(structured_sample)
        #     textual_data.append(textual_sample)
        # classes = np.random.randint(1, size=(10)).astype('float32')
        # structured_data = np.asarray(structured_data)
        # textual_data = np.asarray(textual_data)
        # print(structured_data)
        # print(textual_data)
        # print(classes)
        # model = kerasAdapter.model
        # # data = {'structured':structured_data, 'textual': textual_data}
        # data, classes = dataTrainGenerator[0]
        # # data['structured'] = data['structured'].astype('float32')
        # # data['textual'] = data['textual'].astype('float32')
        # # classes = classes.astype('float32')
        # print("##################################################")
        # print(data['structured'], data['structured'].shape)
        # print(structured_data, structured_data.shape)
        # print("==========================")
        # print(data['textual'], data['textual'].shape)
        # print(textual_data, textual_data.shape)
        # input()
        # model.fit(data, classes, epochs=2)
        # exit()

        kerasAdapter.fit(dataTrainGenerator, epochs=epochs, callbacks=None, class_weights=None, use_multiprocessing=False)
        kerasAdapter.save(trained_model_path)
    else:
        kerasAdapter = KerasAdapter.load_model(trained_model_path)

    metrics = test_model(kerasAdapter, dataTestGenerator, fold)
    exit()
    metrics = pandas.Series(metrics)
    all_metrics.append(metrics)
    all_metrics.to_csv(results_file_path)

    end = datetime.datetime.now()
    time_to_train = end - start
    hours, remainder = divmod(time_to_train.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    print_with_time('Took {:02}:{:02}:{:02} to train the level zero models for structured data'.format(int(hours), int(minutes), int(seconds)))
    fold += 1