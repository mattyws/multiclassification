import csv
import os
import pickle

import pandas as pd
import numpy as np

import keras

from sklearn.model_selection._split import StratifiedKFold, train_test_split
from tensorflow.keras.metrics import AUC

from adapter import KerasAdapter
from resources.data_representation import TransformClinicalTextsRepresentations
from multiclassification.parameters.classification_parameters import timeseries_textual_training_parameters as parameters
from multiclassification.parameters.classification_parameters import model_tuner_parameters as tuner_parameters

from resources import functions
from resources.data_generators import LengthLongitudinalDataGenerator
from resources.functions import test_model, print_with_time, whitespace_tokenize_text, train_representation_model
from resources.keras_callbacks import Metrics
from resources.model_creators import MultilayerKerasRecurrentNNCreator, MultilayerTemporalConvolutionalNNCreator, \
    KerasTunerModelCreator, MultilayerTemporalConvolutionalNNHyperModel
import kerastuner as kt

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
DATETIME_PATTERN = "%Y-%m-%d %H:%M:%S"

problem = 'mortality'
training_base_directory = parameters['multiclassification_base_path'] + parameters['training_directory_path']
training_directory = training_base_directory + parameters[problem+"_directory"] \
                     + parameters['execution_saving_path']
checkpoint_directory = training_directory + parameters['training_checkpoint']
if not os.path.exists(checkpoint_directory):
    os.makedirs(checkpoint_directory)
with open(checkpoint_directory + parameters['execution_parameters_filename'], 'wb') as handler:
    pickle.dump(parameters, handler)

# Loading csv
print_with_time("Loading data")
dataset_path = parameters['multiclassification_base_path'] + parameters[problem+'_directory'] \
               + parameters[problem+'_dataset_csv']

data_csv = pd.read_csv(dataset_path)
data_csv = data_csv.sort_values(['episode'])

print_with_time("Class distribution")
print(data_csv['label'].value_counts())


# Using a seed always will get the same data split even if the training stops
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=15)


print_with_time("Training/Loading representation model")
embedding_size = parameters['textual_embedding_size']
min_count = parameters['textual_min_count']
workers = parameters['textual_workers']
window = parameters['textual_window']
iterations = parameters['textual_iterations']
textual_input_shape = (None, embedding_size)
preprocessing_pipeline = [whitespace_tokenize_text]

texts_hourly_merged_dir = parameters['multiclassification_base_path'] + "textual_hourly_merged/"
representation_model_data = [texts_hourly_merged_dir + x for x in os.listdir(texts_hourly_merged_dir)]
textual_representation_path = os.path.join(parameters['textual_representation_model_path'], str(embedding_size))
textual_representation_model_path = os.path.join(textual_representation_path,
                                                    parameters['textual_representation_model_filename'])
if not os.path.exists(textual_representation_path):
    os.makedirs(textual_representation_path)
representation_model = train_representation_model(representation_model_data,
                                                textual_representation_model_path,
                                                  min_count, embedding_size, workers, window, iterations,
                                                  hs=parameters['textual_doc2vec_hs'], dm=parameters['textual_doc2vec_dm'],
                                                  negative=parameters['textual_doc2vec_negative'],
                                                  preprocessing_pipeline=preprocessing_pipeline, word2vec=False)


print_with_time("Transforming/Retrieving representation")
notes_textual_representation_path = os.path.join(textual_representation_path, problem,
                                                 parameters['notes_textual_representation_directory'])
if not os.path.exists(notes_textual_representation_path):
    os.makedirs(notes_textual_representation_path)
texts_transformer = TransformClinicalTextsRepresentations(representation_model, embedding_size=embedding_size,
                                                          window=window,
                                                          representation_save_path=notes_textual_representation_path,
                                                          is_word2vec=False)
representation_model = None
new_paths = texts_transformer.transform(data_csv, 'textual_path', preprocessing_pipeline=preprocessing_pipeline,
                                        remove_temporal_axis=parameters['remove_temporal_axis'],
                                        remove_no_text_constant=parameters['remove_no_text_constant'])
classes = np.asarray(new_paths['label'].tolist())
data = np.asarray(new_paths['path'].tolist())


def get_shape(lst, shape=()):
    """
    returns the shape of nested lists similarly to numpy's shape.

    :param lst: the nested list
    :param shape: the shape up to the current recursion depth
    :return: the shape including the current depth
            (finally this will be the full depth)
    """

    if not isinstance(lst, np.ndarray):
        # base case
        print(type(lst))
        return shape

    # peek ahead and assure all lists in the next depth
    # have the same length
    if isinstance(lst[0], np.ndarray):
        l = len(lst[0])
        if not all(len(item) == l for item in lst):
            msg = 'not all lists have the same length'
            raise ValueError(msg)

    shape += (len(lst), )

    # recurse
    shape = get_shape(lst[0], shape)

    return shape

# consumed = 0
# total_files = len(data)
# for path in data:
#     sys.stderr.write('\rdone {0:%}'.format(consumed / total_files))
#     doc = pickle.load(open(path, 'rb'))
#     new_doc = []
#     for sentence in doc:
#         sentence = np.squeeze(sentence)
#         print(get_shape(sentence))
#         new_doc.append(sentence)
#     new_doc = np.asarray(new_doc)
#     with open(path, 'wb') as f :
#          pickle.dump(new_doc, f)
#     consumed += 1


# Get input shape
aux = pickle.load(open(data[0], 'rb'))
print(get_shape(aux))
aux = np.asarray(aux)
inputShape = (None, len(aux[0]))
# parameters['model_tunning'] = False
if parameters['model_tunning']:
    training_samples_path = training_directory + parameters['training_samples_filename']
    training_classes_path = training_directory + parameters['training_classes_filename']
    optimization_samples_path = training_directory + parameters['optimization_samples_filename']
    optimization_classes_path = training_directory + parameters['optimization_classes_filename']

    if not os.path.exists(training_samples_path):
        data, data_opt, classes, classes_opt = train_test_split(data, classes, stratify=classes,
                                                                test_size=parameters['optimization_split_rate'])
        with open(training_samples_path, 'wb') as f:
            pickle.dump(data, f)
        with open(training_classes_path, 'wb') as f:
            pickle.dump(classes, f)
        with open(optimization_samples_path, 'wb') as f:
            pickle.dump(data_opt, f)
        with open(optimization_classes_path, 'wb') as f:
            pickle.dump(classes_opt, f)
    else:
        with open(training_samples_path, 'rb') as f:
            data = pickle.load(f)
        with open(training_classes_path, 'rb') as f:
            classes = pickle.load(f)
        with open(optimization_samples_path, 'rb') as f:
            data_opt = pickle.load(f)
        with open(optimization_classes_path, 'rb') as f:
            classes_opt = pickle.load(f)

    print_with_time("Creating optimization generators")
    training_events_sizes_file_path = training_directory + parameters['training_events_sizes_filename'].format('opt')
    training_events_sizes_labels_file_path = training_directory + parameters[
        'training_events_sizes_labels_filename'].format('opt')
    testing_events_sizes_file_path = training_directory + parameters['testing_events_sizes_filename'].format('opt')
    testing_events_sizes_labels_file_path = training_directory + parameters[
        'testing_events_sizes_labels_filename'].format('opt')

    # print(data_opt)
    # print(classes_opt)
    # print(training_events_sizes_file_path)
    # print(training_events_sizes_labels_file_path)
    # exit()
    train_sizes, train_labels = functions.divide_by_events_lenght(data_opt
                                                                  , classes_opt
                                                                  , sizes_filename=training_events_sizes_file_path
                                                                  ,
                                                                  classes_filename=training_events_sizes_labels_file_path)
    dataTrainGenerator = LengthLongitudinalDataGenerator(train_sizes, train_labels,
                                                         max_batch_size=parameters['batchSize'])
    dataTrainGenerator.create_batches()

    # model_builder = MultilayerKerasRecurrentNNHyperModel(inputShape,
    #                                                     parameters['numOutputNeurons'],
    #                                                     True,
    #                                                     [AUC(name='auc')],
    #                                                     tuner_parameters)

    model_builder = MultilayerTemporalConvolutionalNNHyperModel(inputShape, parameters['numOutputNeurons'],
                                                                [AUC()], tuner_parameters)
    tunning_directory = checkpoint_directory + parameters['tunning_directory']
    tuner = kt.Hyperband(model_builder,
                         objective=kt.Objective('auc', direction="max"),
                         max_epochs=40,
                         directory=tunning_directory,
                         project_name='timeseries',
                         factor=4)
    tuner.search(dataTrainGenerator, epochs=40)
    modelCreator = KerasTunerModelCreator(tuner, model_builder.name)
else:
    if not parameters['tcn']:
        modelCreator = MultilayerKerasRecurrentNNCreator(inputShape, parameters['outputUnits'],
                                                         parameters['numOutputNeurons'],
                                                         loss=parameters['loss'],
                                                         layersActivations=parameters['layersActivations'],
                                                         networkActivation=parameters['networkActivation'],
                                                         gru=parameters['gru'], use_dropout=parameters['useDropout'],
                                                         dropout=parameters['dropout'], kernel_regularizer=None,
                                                         metrics=[keras.metrics.binary_accuracy],
                                                         optimizer=parameters['optimizer'])
    else:
        modelCreator = MultilayerTemporalConvolutionalNNCreator(inputShape, parameters['outputUnits'],
                                                                parameters['numOutputNeurons'],
                                                                loss=parameters['loss'],
                                                                layersActivations=parameters['layersActivations'],
                                                                networkActivation=parameters['networkActivation'],
                                                                pooling=parameters['pooling'],
                                                                kernel_sizes=parameters['kernel_sizes'],
                                                                use_dropout=parameters['useDropout'],
                                                                dilations=parameters['dilations'],
                                                                nb_stacks=parameters['nb_stacks'],
                                                                dropout=parameters['dropout'], kernel_regularizer=None,
                                                                metrics=[keras.metrics.binary_accuracy],
                                                                optimizer=parameters['optimizer'])

i = 0
# ====================== Script that start training new models
result_file_path = checkpoint_directory + parameters['result_filename']
with open(result_file_path, 'a+') as cvsFileHandler: # where the results for each fold are appended
    dictWriter = None
    for trainIndex, testIndex in kf.split(data, classes):
        trained_model_path = checkpoint_directory + parameters['trained_model_filename'].format(i)
        # if os.path.exists(trained_model_path):
        #     print("Pass fold {}".format(i))
        #     i += 1
        #     continue
        print_with_time("Fold {}".format(i))
        print_with_time("Creating generators")
        # dataTrainGenerator = LongitudinalDataGenerator(normalized_data[trainIndex],
        #                                                classes[trainIndex], parameters['batchSize'])
        # dataTestGenerator = LongitudinalDataGenerator(normalized_data[testIndex],
        #                                               classes[testIndex], parameters['batchSize'])

        training_events_sizes_file_path = training_directory + parameters['training_events_sizes_filename'].format(i)
        training_events_sizes_labels_file_path = training_directory + parameters['training_events_sizes_labels_filename'].format(i)
        testing_events_sizes_file_path = training_directory + parameters['testing_events_sizes_filename'].format(i)
        testing_events_sizes_labels_file_path = training_directory + parameters['testing_events_sizes_labels_filename'].format(i)

        train_sizes, train_labels = functions.divide_by_events_lenght(data[trainIndex]
                                                                      , classes[trainIndex]
                                                                      , sizes_filename=training_events_sizes_file_path
                                                                      , classes_filename=training_events_sizes_labels_file_path)
        test_sizes, test_labels = functions.divide_by_events_lenght(data[testIndex], classes[testIndex]
                                                                    , sizes_filename = testing_events_sizes_file_path
                                                                    , classes_filename = testing_events_sizes_labels_file_path)

        dataTrainGenerator = LengthLongitudinalDataGenerator(train_sizes, train_labels, max_batch_size=parameters['batchSize'])
        dataTrainGenerator.create_batches()
        dataTestGenerator = LengthLongitudinalDataGenerator(test_sizes, test_labels, max_batch_size=parameters['batchSize'])
        dataTestGenerator.create_batches()
        if not os.path.exists(trained_model_path):
            kerasAdapter = modelCreator.create(model_summary_filename=checkpoint_directory+'model_summary.txt')
            epochs = parameters['trainingEpochs']
            metrics_callback = Metrics(dataTestGenerator)
            print_with_time("Training model")
            kerasAdapter.fit(dataTrainGenerator, epochs=epochs, callbacks=None, class_weights=None, use_multiprocessing=False)
            kerasAdapter.save(trained_model_path)
        else:
            kerasAdapter = KerasAdapter.load_model(trained_model_path)
        print_with_time("Testing model")
        metrics = test_model(kerasAdapter, dataTestGenerator, i)
        if dictWriter is None:
            dictWriter = csv.DictWriter(cvsFileHandler, metrics.keys())
        if metrics['fold'] == 0:
            dictWriter.writeheader()
        dictWriter.writerow(metrics)
        i += 1

# Evaluating k-fold
results = pd.read_csv(result_file_path)
print(results.describe())
