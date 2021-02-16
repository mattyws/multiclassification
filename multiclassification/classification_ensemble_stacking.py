import csv
import datetime
import json
import os
import pickle
from collections import Counter
from multiprocessing import set_start_method

import pandas
import pandas as pd
import numpy as np
import gc

import keras
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

from multiclassification.ensemble_kappa_agreement import kappa_aggreement
from multiclassification.parameters.classification_parameters import ensemble_parameters as parameters

from resources import functions
from adapter import KerasAdapter
from resources.data_generators import LengthLongitudinalDataGenerator, LongitudinalDataGenerator, \
    ArrayDataGenerator
from resources.data_representation import EnsembleMetaLearnerDataCreator, TransformClinicalTextsRepresentations
from ensemble_training import TrainEnsembleAdaBoosting, TrainEnsembleBagging, split_classes
from resources.functions import test_model, print_with_time, escape_invalid_xml_characters, \
    escape_html_special_entities, \
    text_to_lower, remove_sepsis_mentions, remove_only_special_characters_tokens, whitespace_tokenize_text, \
    train_representation_model, remove_columns_for_classification, remove_empty_textual_data_episodes
from resources.model_creators import MultilayerKerasRecurrentNNCreator, EnsembleModelCreator, \
    MultilayerTemporalConvolutionalNNCreator, NoteeventsClassificationModelCreator, KerasTunerModelCreator, \
    MultilayerTemporalConvolutionalNNHyperModel, MultilayerKerasRecurrentNNHyperModel
from resources.normalization import Normalization, NormalizationValues
import kerastuner as kt

from result_evaluation import ModelEvaluation
import os

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
    models = []
    # models.append(kerasAdapter)
    models.append(LinearSVC())
    models.append(LogisticRegression())
    models.append(GaussianNB())
    models.append(DecisionTreeClassifier())
    models.append(MLPClassifier(max_iter=epochs))
    start = datetime.datetime.now()
    class_weights = class_weight.compute_class_weight('balanced',
                                                      np.unique(classes),
                                                      classes)
    mapped_weights = dict()
    for value in np.unique(classes):
        mapped_weights[value] = class_weights[value]
    class_weights = mapped_weights
    if not parameters['use_class_weight']:
        class_weights = None
    for model in models:
        if isinstance(model, KerasAdapter):
            dataGenerator = ArrayDataGenerator(data, classes, parameters['meta_learner_batch_size'])
            kerasAdapter.fit(dataGenerator, epochs=epochs, use_multiprocessing=False, class_weights=class_weights)
        else:
            model.fit(data, classes)
    return models

def test_meta_model_on_data(models, data, data_columns, data_origin):
    results = []
    results_prediction_score = []
    for model in models:
        model_evaluation = test_sklearn_meta_models_on_generator(model, data, data_columns)
        result = model_evaluation.metrics
        for f, p, c in zip(model_evaluation.files, model_evaluation.predictions_scores, model_evaluation.predictions_classes):
            results_prediction_score.append({'episode':f, 'probas':p, 'classes':c, 'model':model.__class__.__name__, "origin":data_origin})
        result['origin'] = data_origin
        result['model'] = model.__class__.__name__
        results.append(result)
    return results, results_prediction_score


def test_sklearn_meta_models_on_generator(model, data:pandas.DataFrame, data_columns) -> ModelEvaluation:
    predicted = []
    trueClasses = []
    files = []
    i = 0
    testing_data = data.loc[:, data_columns].values
    try:
        positive = np.argmax(model.classes_)
        r = model.predict_proba(testing_data)
        if len(r[0]) > 1:
            new_r = []
            for probas in r:
                new_r.append(probas[positive])
            r = np.asarray(new_r)
    except:
        try:
            r = model.decision_function(testing_data)
            r = (r - r.min()) / (r.max() - r.min())
        except:
            r = model.predict(testing_data)
    r = r.flatten()
    predicted = r
    trueClasses = data.loc[:, 'label']
    files = data.loc[:, 'episode']
    # for i in range(len(generator)):
    #     sys.stderr.write('\rdone {0:%}'.format(i / len(generator)))
    #     data = generator[i]
    #     try:
    #         positive = np.argmax(model.classes_)
    #         r = model.predict_proba(data[0])
    #         if len(r[0]) > 1:
    #             new_r = []
    #             for probas in r:
    #                 new_r.append(probas[positive])
    #             r = np.asarray(new_r)
    #     except:
    #         try:
    #             r = model.decision_function(data[0])
    #             if len(r[0]) > 1:
    #                 new_r = []
    #                 for probas in r:
    #                     new_r.append(probas[positive])
    #                 r = np.asarray(new_r)
    #         except:
    #             r = model.predict(data[0])
    #     r = r.flatten()
    #     predicted.extend(r)
    #     trueClasses.extend(data[1])
    #     files.extend(generator.batches[i])
    evaluation =  ModelEvaluation(model, files, trueClasses, predicted)
    return evaluation

if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    DATETIME_PATTERN = "%Y-%m-%d %H:%M:%S"
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
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
    dataset_path = os.path.join(parameters['multiclassification_base_path'], parameters[problem+'_directory'],
                                parameters[problem+'_dataset_csv'])

    data_csv = pd.read_csv(dataset_path)
    data_csv = data_csv.sort_values(['episode'])
    data_csv.loc[:, 'structured_path'] = data_csv['structured_path'].apply(lambda x : os.path.join(parameters['multiclassification_base_path'], x))
    data_csv.loc[:, 'textual_path'] = data_csv['textual_path'].apply(lambda x : os.path.join(parameters['multiclassification_base_path'], x))

    if parameters['remove_no_text_constant']:
        data_csv = remove_empty_textual_data_episodes(data_csv, 'textual_path')
    print(data_csv['label'].value_counts())

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
    textual_representation_model_path = os.path.join(parameters['textual_representation_model_path'], str(embedding_size),
                                                        parameters['textual_representation_model_filename'])
    if not os.path.exists(textual_representation_model_path):
        os.makedirs(textual_representation_model_path)
    representation_model = train_representation_model(representation_model_data,
                                                    textual_representation_model_path,
                                                      min_count, embedding_size, workers, window, iterations,
                                                      hs=parameters['textual_doc2vec_hs'], dm=parameters['textual_doc2vec_dm'],
                                                      negative=parameters['textual_doc2vec_negative'],
                                                      preprocessing_pipeline=preprocessing_pipeline, word2vec=False)

    print_with_time("Transforming/Retrieving representation")
    notes_textual_representation_path = os.path.join(parameters['textual_representation_model_path'],
                                                     str(embedding_size), problem,
                                                     parameters['notes_textual_representation_directory'])
    if not os.path.exists(notes_textual_representation_path):
        os.makedirs(notes_textual_representation_path)
    texts_transformer = TransformClinicalTextsRepresentations(representation_model, embedding_size=embedding_size,
                                                              window=window,
                                                              representation_save_path=notes_textual_representation_path,
                                                              is_word2vec=False)
    texts_transformer.transform(data_csv, 'textual_path', preprocessing_pipeline=preprocessing_pipeline,
                                remove_no_text_constant=parameters['remove_no_text_constant'])
    texts_transformer.clear()
    # Free memory space as when us multiprocessing it don't copy the doc2vec model onto the other processes
    del representation_model
    gc.collect()
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

    structured_model_creator = []
    textual_model_creator = []

    if parameters['model_tunning']:
        print_with_time("Tunning hyperparameters")
        optimization_samples_path = training_directory + parameters['optimization_samples_filename']
        optimization_classes_path = training_directory + parameters['optimization_classes_filename']
        from multiclassification.parameters.classification_parameters import timeseries_tuning_parameters, textual_tuning_parameters
        optimization_smodel_hypermodels = []
        optimization_smodel_hypermodels.append(MultilayerTemporalConvolutionalNNHyperModel(structured_input_shape,
                                                                   parameters['structured_output_neurons'],
                                                                    [AUC(name='auc')], timeseries_tuning_parameters))
        optimization_smodel_hypermodels.append(MultilayerKerasRecurrentNNHyperModel(structured_input_shape,
                                                                   parameters['structured_output_neurons'],
                                                                    parameters['structured_gru'],
                                                                   [AUC(name='auc')],
                                                                   timeseries_tuning_parameters))
        optimization_tmodel_hypermodels = []
        optimization_tmodel_hypermodels.append(MultilayerTemporalConvolutionalNNHyperModel(textual_input_shape,
                                                                                           parameters['textual_output_neurons'],
                                                                                           [AUC(name='auc')],
                                                                                           textual_tuning_parameters))
        optimization_tmodel_hypermodels.append(MultilayerKerasRecurrentNNHyperModel(textual_input_shape,
                                                                                    parameters['textual_output_neurons'],
                                                                                    parameters['textual_gru'],
                                                                                    [AUC(name='auc')],
                                                                                    textual_tuning_parameters))
        if not os.path.exists(training_samples_path):
            len_dataset = len(episodes)
            print("Dataset size: ", len_dataset)
            print("Test split rate: ",parameters['train_test_split_rate'])
            print("Optimization split rate: ", parameters['optimization_split_rate'])
            len_evaluation_dataset = int(len_dataset * parameters['train_test_split_rate'])
            len_optimization_dataset = int(len_dataset * parameters['optimization_split_rate'])
            print("Test dataset len: ", len_evaluation_dataset)
            print("Optimization dataset len: ", len_optimization_dataset)
            X, X_val, classes, classes_evaluation = train_test_split(episodes, classes, stratify=classes,
                                                                     test_size=len_evaluation_dataset)
            print("Após split de teste", Counter(classes))
            print("Contagem de dados de teste", Counter(classes_evaluation))

            X, X_opt, classes, classes_opt = train_test_split(X, classes, stratify=classes,
                                                              test_size=len_optimization_dataset)
            print("Distribuição após split dos dados de otimização")
            print(Counter(classes))
            print("Distribuição de classes dados de otimização:", Counter(classes_opt))

            if parameters['balance_training_data']:
                #############################################
                ### Balancing instances on trainning data ###
                #############################################
                aux = data_csv[data_csv['episode'].isin(X)]
                len_positive = len(aux[aux['label'] == 1])
                subsample = aux[aux['label'] == 0].sample(len_positive)
                subsample = subsample.append(aux[aux['label'] == 1])
                print("Distribuição dados de treinamento depois do balanceamento:", subsample['label'].value_counts())
                X = subsample['episode'].tolist()
                classes = subsample['label'].tolist()
                #############################################
                ### Balancing instances on optimization data ###
                #############################################
                aux = data_csv[data_csv['episode'].isin(X_opt)]
                len_positive = len(aux[aux['label'] == 1])
                subsample = aux[aux['label'] == 0].sample(len_positive)
                subsample = subsample.append(aux[aux['label'] == 1])
                print("Distribuição dados de otimização depois do balanceamento:", subsample['label'].value_counts())
                X_opt = subsample['episode'].tolist()
                classes_opt = subsample['label'].tolist()
            with open(training_samples_path, 'wb') as f:
                pickle.dump(X, f)
            with open(training_classes_path, 'wb') as f:
                pickle.dump(classes, f)
            with open(testing_samples_path, 'wb') as f:
                pickle.dump(X_val, f)
            with open(testing_classes_path, 'wb') as f:
                pickle.dump(classes_evaluation, f)
            with open(optimization_samples_path, 'wb') as f :
                pickle.dump(X_opt, f)
            with open(optimization_classes_path, 'wb') as f:
                pickle.dump(classes_opt, f)
        else:
            with open(training_samples_path, 'rb') as f:
                X = pickle.load(f)
            with open(training_classes_path, 'rb') as f:
                classes = pickle.load(f)
            with open(testing_samples_path, 'rb') as f:
                X_val = pickle.load(f)
            with open(testing_classes_path, 'rb') as f:
                classes_evaluation = pickle.load(f)
            with open(optimization_samples_path, 'rb') as f :
                X_opt = pickle.load(f)
            with open(optimization_classes_path, 'rb') as f:
                classes_opt = pickle.load(f)

        optimization_df = data_csv[data_csv['episode'].isin(X_opt)]
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

        for hypermodel in optimization_smodel_hypermodels:
            tunning_directory = checkpoint_directory + parameters['tunning_directory']
            tuner = kt.Hyperband(hypermodel,
                                 objective=kt.Objective('auc', direction="max"),
                                 max_epochs=10,
                                 directory=tunning_directory,
                                 project_name=hypermodel.name,
                                 factor=3)
            tuner.search(dataTrainGenerator, epochs=10)
            modelCreator = KerasTunerModelCreator(tuner, hypermodel.name)
            structured_model_creator.append(modelCreator)

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
        for hypermodel in optimization_tmodel_hypermodels:
            hypermodel.name = hypermodel.name + '_textual'
            tunning_directory = checkpoint_directory + parameters['tunning_directory']
            tuner = kt.Hyperband(hypermodel,
                                 objective=kt.Objective('auc', direction="max"),
                                 max_epochs=10,
                                 directory=tunning_directory,
                                 project_name=hypermodel.name,
                                 factor=3)
            tuner.search(dataTrainGenerator, epochs=10)
            modelCreator = KerasTunerModelCreator(tuner, hypermodel.name)
            textual_model_creator.append(modelCreator)


    else :
        if not os.path.exists(training_samples_path):
            X, X_val, classes, classes_evaluation = train_test_split(episodes, classes, stratify=classes,
                                                                     test_size=parameters['train_test_split_rate'])
            with open(training_samples_path, 'wb') as f:
                pickle.dump(X, f)
            with open(training_classes_path, 'wb') as f:
                pickle.dump(classes, f)
            with open(testing_samples_path, 'wb') as f:
                pickle.dump(X_val, f)
            with open(testing_classes_path, 'wb') as f:
                pickle.dump(classes_evaluation, f)
        else:
            with open(training_samples_path, 'rb') as f:
                X = pickle.load(f)
            with open(training_classes_path, 'rb') as f:
                classes = pickle.load(f)
            with open(testing_samples_path, 'rb') as f:
                X_val = pickle.load(f)
            with open(testing_classes_path, 'rb') as f:
                classes_evaluation = pickle.load(f)

        structured_model_creator.append(MultilayerKerasRecurrentNNCreator(structured_input_shape,
                                                                parameters['structured_output_units'],
                                                                 parameters['structured_output_neurons'],
                                                                 loss=parameters['structured_loss'],
                                                                 layersActivations=parameters['structured_layers_activations'],
                                                                 networkActivation=parameters['structured_network_activation'],
                                                                 gru=parameters['structured_gru'],
                                                                  kernel_regularizer=None,
                                                                 use_dropout=parameters['structured_use_dropout'],
                                                                dropout=parameters['structured_dropout'],
                                                                 metrics=[keras.metrics.binary_accuracy, AUC()],
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
                                                                kernel_regularizer=None,
                                                                metrics=[keras.metrics.binary_accuracy, AUC()],
                                                                optimizer=parameters['structured_optimizer']))
        textual_model_creator.append(MultilayerTemporalConvolutionalNNCreator(textual_input_shape, parameters['textual_output_units'],
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
                                                                metrics=[keras.metrics.binary_accuracy, AUC()],
                                                                optimizer=parameters['textual_optimizer']))
        structured_model_creator.append(MultilayerKerasRecurrentNNCreator(structured_input_shape,
                                                                          parameters['textual_output_units'],
                                                                          parameters['textual_output_neurons'],
                                                                          loss=parameters['textual_loss'],
                                                                          layersActivations=parameters[
                                                                              'textual_layers_activations'],
                                                                          networkActivation=parameters[
                                                                              'textual_network_activation'],
                                                                          gru=parameters['textual_gru'],
                                                                          kernel_regularizer=None,
                                                                          use_dropout=parameters['textual_use_dropout'],
                                                                          dropout=parameters['textual_dropout'],
                                                                          metrics=[keras.metrics.binary_accuracy, AUC()],
                                                                          optimizer=parameters['textual_optimizer']))

    training_df = data_csv[data_csv['episode'].isin(X)]
    evaluation_df = data_csv[data_csv['episode'].isin(X_val)]

    classes = np.asarray(training_df['label'].tolist())
    classes_evaluation = np.asarray(evaluation_df['label'].tolist())

    structured_data = np.asarray(training_df['structured_path'].tolist())
    structured_evaluation = np.asarray(evaluation_df['structured_path'].tolist())
    textual_data = np.asarray(training_df['textual_path'].tolist())
    textual_evaluation = np.asarray(evaluation_df['textual_path'].tolist())

    textual_transformed_data = np.asarray(texts_transformer.get_new_paths(textual_data))
    textual_evaluation = np.asarray(texts_transformer.get_new_paths(textual_evaluation))



    # structured_data, structured_evaluation = train_evaluation_split_with_icustay(X, structured_data)
    # textual_transformed_data, textual_evaluation = train_evaluation_split_with_icustay(X, textual_transformed_data)

    print(structured_data)
    print(len(structured_data), len(classes))
    print(len(structured_evaluation), len(classes_evaluation))
    print(len(textual_transformed_data), len(classes))
    print(len(textual_evaluation), len(classes_evaluation))

    (unique, counts) = np.unique(classes, return_counts=True)
    frequencies = np.asarray((unique, counts)).T
    print("Frequency of classes in training")
    print(frequencies)
    (unique, counts) = np.unique(classes_evaluation, return_counts=True)
    frequencies = np.asarray((unique, counts)).T
    print("Frequency of classes in evaluation")
    print(frequencies)

    eval_normalization_values_path = training_directory + parameters['evaluation_normalization_values_filename']
    eval_values = normalization_values.get_normalization_values(evaluation_df['structured_path'].tolist(),
                                                                saved_file_name=eval_normalization_values_path)
    eval_normalization_temporary_data_path = training_directory + parameters[
        'evaluation_normalization_temporary_data_directory']
    eval_normalizer = Normalization(eval_values, temporary_path=eval_normalization_temporary_data_path)
    print_with_time("Normalizing evaluation data")
    eval_normalizer.normalize_files(evaluation_df['structured_path'].tolist())
    eval_normalized_data = np.array(eval_normalizer.get_new_paths(evaluation_df['structured_path'].tolist()))


    evaluation_sevents_sizes_file_path = training_directory \
                                      + parameters['structured_training_events_sizes_filename'].format('seval')
    evaluation_sevents_sizes_labels_file_path = training_directory \
                                             + parameters['structured_training_events_sizes_labels_filename'].format('seval')

    evaluation_sizes, evaluation_labels = functions.divide_by_events_lenght(eval_normalized_data
                                                                            , classes_evaluation
                                                                            ,
                                                                            sizes_filename=evaluation_sevents_sizes_file_path
                                                                            ,
                                                                            classes_filename=evaluation_sevents_sizes_labels_file_path)
    evaluationStructuredTSGenerator = LengthLongitudinalDataGenerator(evaluation_sizes, evaluation_labels,
                                                          max_batch_size=parameters['structured_batch_size'])
    evaluationStructuredTSGenerator.create_batches()


    evaluation_textual_transformed_data = np.asarray(texts_transformer.get_new_paths(evaluation_df['textual_path'].tolist()))
    evaluation_tevents_sizes_file_path = training_directory \
                                      + parameters['textual_training_events_sizes_filename'].format('teval')
    evaluation_tevents_sizes_labels_file_path = training_directory \
                                             + parameters['textual_training_events_sizes_labels_filename'].format(
        'teval')

    evaluation_sizes, evaluation_labels = functions.divide_by_events_lenght(evaluation_textual_transformed_data
                                                                  , classes_evaluation
                                                                  , sizes_filename=evaluation_tevents_sizes_file_path
                                                                  ,
                                                                  classes_filename=evaluation_tevents_sizes_labels_file_path)

    evaluationTextualTSGenerator = LengthLongitudinalDataGenerator(evaluation_sizes, evaluation_labels,
                                                         max_batch_size=parameters['textual_batch_size'])
    evaluationTextualTSGenerator.create_batches()


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
    results_file_path = checkpoint_directory + parameters['metrics_filename']
    level_zero_results_file_path = checkpoint_directory + parameters['level_zero_result_filename']

    dictWriter = None
    level_zero_dict_writer = None
    for trainIndex, testIndex in kf.split(structured_data, classes):
        print(len(trainIndex), len(testIndex))
        print(len(structured_data[trainIndex]), len(structured_data[testIndex]))
        print(len(textual_transformed_data[trainIndex]), len(textual_transformed_data[testIndex]))
        print(len(classes[trainIndex]), len(classes[testIndex]))

        fold_structured_predictions_path = checkpoint_directory + parameters['fold_structured_predictions_filename']\
            .format(fold)
        fold_structured_representations_path = checkpoint_directory + parameters['fold_structured_representations_filename']\
            .format(fold)
        fold_textual_predictions_path = checkpoint_directory + parameters['fold_textual_predictions_filename'] \
            .format(fold)
        fold_textual_representations_path = checkpoint_directory + parameters['fold_textual_representations_filename'] \
            .format(fold)
        fold_metrics_path = checkpoint_directory + parameters['fold_metrics_filename'].format(fold)

        if parameters['balance_training_data']:
            fold_evaluation_metrics_path = checkpoint_directory + parameters['fold_evaluation_metrics_filename'].format(fold)

        if os.path.exists(fold_metrics_path):
            print("Pass fold {}".format(fold))

            fold_structured_predictions = pandas.read_csv(fold_structured_predictions_path, index_col=0)
            fold_structured_representations= pandas.read_csv(fold_structured_representations_path, index_col=0)
            fold_textual_predictions= pandas.read_csv(fold_textual_predictions_path, index_col=0)
            fold_textual_representations= pandas.read_csv(fold_textual_representations_path, index_col=0)
            fold_metrics = pandas.read_csv(fold_metrics_path)

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
        if not parameters['use_class_weight']:
            class_weights = None
        fold_structured_predictions = dict()
        fold_structured_representations = dict()
        fold_textual_predictions = dict()
        fold_textual_representations = dict()

        fold_predictions = dict()
        fold_representations = dict()
        fold_metrics = []
        fold_evaluation_metrics = []

        structured_ensemble = None
        model_adapters = []
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

        training_events_sizes_file_path = training_directory \
                                          + parameters['structured_training_events_sizes_filename'].format(fold)
        training_events_sizes_labels_file_path = training_directory \
                                          + parameters['structured_training_events_sizes_labels_filename'].format(fold)
        testing_events_sizes_file_path = training_directory \
                                         + parameters['structured_testing_events_sizes_filename'].format(fold)
        testing_events_sizes_labels_file_path = training_directory \
                                         + parameters['structured_testing_events_sizes_labels_filename'].format(fold)

        train_sizes, train_labels = functions.divide_by_events_lenght(normalized_data[trainIndex]
                                                                      , classes[trainIndex]
                                                                      , sizes_filename=training_events_sizes_file_path
                                                                      , classes_filename=training_events_sizes_labels_file_path)
        test_sizes, test_labels = functions.divide_by_events_lenght(normalized_data[testIndex], classes[testIndex]
                                                                    , sizes_filename=testing_events_sizes_file_path
                                                                    , classes_filename=testing_events_sizes_labels_file_path)

        dataTrainGenerator = LengthLongitudinalDataGenerator(train_sizes, train_labels,
                                                             max_batch_size=parameters['structured_batch_size'])
        dataTrainGenerator.create_batches()
        dataTestGenerator = LengthLongitudinalDataGenerator(test_sizes, test_labels,
                                                            max_batch_size=parameters['structured_batch_size'])
        dataTestGenerator.create_batches()

        start = datetime.datetime.now()
        print_with_time("Training level 0 models for structured data")
        for i, model_creator in enumerate(structured_model_creator):
            saved_model_path = checkpoint_directory + parameters['structured_weak_model'].format(model_creator.name, fold)
            if os.path.exists(saved_model_path):
                adapter = KerasAdapter.load_model(saved_model_path)
            else:
                adapter = model_creator.create(model_summary_filename=checkpoint_directory
                                                                      + model_creator.name + '_model_summary.txt')
                adapter.fit(dataTrainGenerator, epochs=parameters['structured_training_epochs'], callbacks=None,
                            class_weights=class_weights, use_multiprocessing=True)
                adapter.save(saved_model_path)
            metrics, results = test_model(adapter, dataTestGenerator, fold, return_predictions=True)
            metrics['model'] = model_creator.name
            fold_metrics.append(metrics)

            if parameters['balance_training_data']:
                eval_metrics = test_model(adapter, evaluationStructuredTSGenerator, fold, return_predictions=False)
                eval_metrics['model'] = model_creator.name
                fold_evaluation_metrics.append(eval_metrics)
            for key in results.keys():
                if not model_creator.name in fold_structured_predictions.keys():
                    fold_structured_predictions[model_creator.name] = dict()
                episode = key.split('/')[-1].split('.')[0]
                fold_structured_predictions[model_creator.name][episode] = results[key]
            model = change_weak_classifiers(adapter.model)
            for i in range(len(dataTestGenerator)):
                sys.stderr.write('\rdone {0:%}'.format(i / len(dataTestGenerator)))
                data = dataTestGenerator[i]
                representations = model.predict(data[0])
                # print(representations)
                for f, r in zip(dataTestGenerator.batches[i], representations):
                    episode = f.split('/')[-1].split('.')[0]
                    if episode not in fold_structured_representations.keys():
                        fold_structured_representations[episode] = []
                    fold_structured_representations[episode].extend(r)


        end = datetime.datetime.now()
        time_to_train = end - start
        hours, remainder = divmod(time_to_train.seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        print_with_time('Took {:02}:{:02}:{:02} to train the level zero models for structured data'.format(int(hours), int(minutes), int(seconds)))

        training_events_sizes_file_path = training_directory \
                                          + parameters['textual_training_events_sizes_filename'].format(fold)
        training_events_sizes_labels_file_path = training_directory \
                                          + parameters['textual_training_events_sizes_labels_filename'].format(fold)
        testing_events_sizes_file_path = training_directory \
                                          + parameters['textual_testing_events_sizes_filename'].format(fold)
        testing_events_sizes_labels_file_path = training_directory \
                                          + parameters['textual_testing_events_sizes_labels_filename'].format(fold)

        train_sizes, train_labels = functions.divide_by_events_lenght(textual_transformed_data[trainIndex]
                                                                      , classes[trainIndex]
                                                                      , sizes_filename=training_events_sizes_file_path
                                                                      , classes_filename=training_events_sizes_labels_file_path)
        test_sizes, test_labels = functions.divide_by_events_lenght(textual_transformed_data[testIndex], classes[testIndex]
                                                                    , sizes_filename=testing_events_sizes_file_path
                                                                    , classes_filename=testing_events_sizes_labels_file_path)
        dataTrainGenerator = LengthLongitudinalDataGenerator(train_sizes, train_labels,
                                                             max_batch_size=parameters['textual_batch_size'])
        dataTrainGenerator.create_batches()
        dataTestGenerator = LengthLongitudinalDataGenerator(test_sizes, test_labels,
                                                            max_batch_size=parameters['textual_batch_size'])
        dataTestGenerator.create_batches()

        print_with_time("Training level 0 models for textual data")
        for i, model_creator in enumerate(textual_model_creator):
            saved_model_path = checkpoint_directory + parameters['textual_weak_model'].format(model_creator.name, fold)
            if os.path.exists(saved_model_path):
                adapter = KerasAdapter.load_model(saved_model_path)
            else:
                start = datetime.datetime.now()
                adapter = model_creator.create()
                adapter.fit(dataTrainGenerator, epochs=parameters['textual_training_epochs'], callbacks=None,
                            class_weights=class_weights, use_multiprocessing=False)
                adapter.save(saved_model_path)
            metrics, results = test_model(adapter, dataTestGenerator, fold, return_predictions=True)
            metrics['model'] = model_creator.name + "_textual"
            fold_metrics.append(metrics)

            if parameters['balance_training_data']:
                eval_metrics = test_model(adapter, evaluationTextualTSGenerator, fold, return_predictions=False)
                eval_metrics['model'] = model_creator.name
                fold_evaluation_metrics.append(eval_metrics)

            for key in results.keys():
                if not model_creator.name + '_textual' in fold_textual_predictions.keys():
                    fold_textual_predictions[model_creator.name + "_textual"] = dict()
                episode = key.split('/')[-1].split('.')[0]
                fold_textual_predictions[model_creator.name + "_textual"][episode] = results[key]
            model = change_weak_classifiers(adapter.model)
            for i in range(len(dataTestGenerator)):
                sys.stderr.write('\rdone {0:%}'.format(i / len(dataTestGenerator)))
                data = dataTestGenerator[i]
                representations = model.predict(data[0])
                # print(representations)
                for f, r in zip(dataTestGenerator.batches[i], representations):
                    episode = f.split('/')[-1].split('.')[0]
                    if episode not in fold_textual_representations.keys():
                        fold_textual_representations[episode] = []
                    fold_textual_representations[episode].extend(r)

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

        fold_structured_representations = fold_structured_representations.add_prefix("s_")
        fold_textual_representations = fold_textual_representations.add_prefix("t_")

        fold_metrics = pandas.DataFrame(fold_metrics)

        fold_structured_predictions.to_csv(fold_structured_predictions_path)
        fold_structured_representations.to_csv(fold_structured_representations_path)
        fold_textual_predictions.to_csv(fold_textual_predictions_path)
        fold_textual_representations.to_csv(fold_textual_representations_path)
        fold_metrics.to_csv(fold_metrics_path)

        if parameters['balance_training_data']:
            fold_evaluation_metrics = pandas.DataFrame(fold_evaluation_metrics)
            fold_evaluation_metrics.to_csv(fold_evaluation_metrics_path)

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
    from scipy.stats import zscore

    if 'Unnamed: 0' in structured_predictions:
        structured_predictions = structured_predictions.set_index(['Unnamed: 0'])
        structured_representations = structured_representations.set_index(['Unnamed: 0'])
        textual_predictions = textual_predictions.set_index(['Unnamed: 0'])
        textual_representations = textual_representations.set_index(['Unnamed: 0'])

    structured_predictions.to_csv(checkpoint_directory + parameters['structured_predictions_filename'])
    structured_representations.to_csv(checkpoint_directory + parameters['structured_representations_filename'])
    textual_predictions.to_csv(checkpoint_directory + parameters['textual_predictions_filename'])
    textual_representations.to_csv(checkpoint_directory + parameters['textual_representations_filename'])
    all_metrics.to_csv(checkpoint_directory + parameters['metrics_filename'])

    # Forcing episodes to be type string
    print(structured_predictions.index.dtype, data_csv['episode'].dtype)
    structured_predictions.index = structured_predictions.index.astype(str)
    structured_representations.index = structured_representations.index.astype(str)
    textual_predictions.index = textual_predictions.index.astype(str)
    textual_representations.index = textual_representations.index.astype(str)
    data_csv.loc[:, 'episode'] = data_csv['episode'].astype(str)
    print(structured_predictions.index.dtype, data_csv['episode'].dtype)

    class_weights = class_weight.compute_class_weight('balanced',
                                                      np.unique(classes),
                                                      classes)
    mapped_weights = dict()
    for value in np.unique(classes):
        mapped_weights[value] = class_weights[value]
    class_weights = mapped_weights
    if not parameters['use_class_weight']:
        class_weights = None

    print(all_metrics)
    metric_mean = all_metrics.groupby(by=['model']).mean()
    print(metric_mean)

    structured_evaluate_predictions = dict()
    structured_evaluate_representations = dict()

    print_with_time("Getting values for normalization")
    normalization_values_path = training_directory + parameters['fold_normalization_values_filename'].format("all")
    normalization_temporary_data_path = training_directory \
                                             + parameters['fold_normalization_temporary_data_directory'].format("all")
    values = normalization_values.get_normalization_values(structured_data, saved_file_name=normalization_values_path)
    normalizer = Normalization(values, temporary_path=normalization_temporary_data_path)

    print_with_time("Normalizing all data")
    normalizer.normalize_files(structured_data)
    normalized_data = np.array(normalizer.get_new_paths(structured_data))
    normalizer.normalize_files(structured_evaluation)
    normalized_evaluation = np.array(normalizer.get_new_paths(structured_evaluation))
    print(normalized_evaluation)
    print(len(normalized_evaluation))

    training_events_sizes_file_path = training_directory \
                                          + parameters['structured_training_events_sizes_filename'].format("all")
    training_events_sizes_labels_file_path = training_directory \
                                      + parameters['structured_training_events_sizes_labels_filename'].format("all")
    testing_events_sizes_file_path = training_directory \
                                     + parameters['structured_testing_events_sizes_filename'].format("all")
    testing_events_sizes_labels_file_path = training_directory \
                                     + parameters['structured_testing_events_sizes_labels_filename'].format("all")

    train_sizes, train_labels = functions.divide_by_events_lenght(normalized_data
                                                                  , classes
                                                                  , sizes_filename=training_events_sizes_file_path
                                                                  , classes_filename=training_events_sizes_labels_file_path)
    test_sizes, test_labels = functions.divide_by_events_lenght(normalized_evaluation, classes_evaluation
                                                                , sizes_filename=testing_events_sizes_file_path
                                                                , classes_filename=testing_events_sizes_labels_file_path)
    dataTrainGenerator = LengthLongitudinalDataGenerator(train_sizes, train_labels,
                                                         max_batch_size=parameters['structured_batch_size'])
    dataTrainGenerator.create_batches()
    dataTestGenerator = LengthLongitudinalDataGenerator(test_sizes, test_labels,
                                                        max_batch_size=parameters['structured_batch_size'])
    dataTestGenerator.create_batches()
    structured_evaluation_predictions = dict()
    structured_evaluation_representations = dict()
    print_with_time("Training structured models on all data")
    for i, model_creator in enumerate(structured_model_creator):
        structured_model_path = checkpoint_directory + parameters['structured_weak_model_all'].format(model_creator.name)
        model_creator.name = model_creator.name + "_all"
        if not os.path.exists(structured_model_path):
            adapter = model_creator.create()
            adapter.fit(dataTrainGenerator, epochs=parameters['structured_training_epochs'], callbacks=None,
                        class_weights=class_weights, use_multiprocessing=False)
            adapter.save(structured_model_path)
        else:
            adapter = KerasAdapter.load_model(structured_model_path)
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
    structured_evaluation_representations.index = structured_evaluation_representations.index.astype(str)

    print_with_time("Training textual models on all data")
    textual_evaluate_predictions = dict()
    textual_evaluate_representations = dict()

    training_events_sizes_file_path = training_directory \
                                      + parameters['textual_training_events_sizes_filename'].format("all")
    training_events_sizes_labels_file_path = training_directory \
                                             + parameters['textual_training_events_sizes_labels_filename'].format("all")
    testing_events_sizes_file_path = training_directory \
                                     + parameters['textual_testing_events_sizes_filename'].format("all")
    testing_events_sizes_labels_file_path = training_directory \
                                            + parameters['textual_testing_events_sizes_labels_filename'].format("all")

    train_sizes, train_labels = functions.divide_by_events_lenght(textual_transformed_data
                                                                  , classes
                                                                  , sizes_filename=training_events_sizes_file_path
                                                                  , classes_filename=training_events_sizes_labels_file_path)
    test_sizes, test_labels = functions.divide_by_events_lenght(textual_evaluation, classes_evaluation
                                                                , sizes_filename=testing_events_sizes_file_path
                                                                , classes_filename=testing_events_sizes_labels_file_path)
    dataTrainGenerator = LengthLongitudinalDataGenerator(train_sizes, train_labels,
                                                         max_batch_size=parameters['structured_batch_size'])
    dataTrainGenerator.create_batches()
    dataTestGenerator = LengthLongitudinalDataGenerator(test_sizes, test_labels,
                                                        max_batch_size=parameters['structured_batch_size'])
    dataTestGenerator.create_batches()
    for i, model_creator in enumerate(textual_model_creator):
        start = datetime.datetime.now()
        textual_model_path = checkpoint_directory + parameters['textual_weak_model_all'].format(model_creator.name)
        model_creator.name = model_creator.name + "_all_textual"
        if not os.path.exists(textual_model_path):
            adapter = model_creator.create()
            adapter.fit(dataTrainGenerator, epochs=parameters['structured_training_epochs'], callbacks=None,
                        class_weights=class_weights, use_multiprocessing=False)
            adapter.save(textual_model_path)
        else:
            adapter = KerasAdapter.load_model(textual_model_path)
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
    textual_evaluate_representations = textual_evaluate_representations.add_prefix("t_")
    textual_evaluate_representations.index = textual_evaluate_representations.index.astype(str)

    ensemble_results = []
    ensemble_predictions = []
    all_predictions = pd.merge(structured_predictions, textual_predictions, left_index=True, right_index=True,
                               how="left")
    # kappas, kappas_positive, kappas_negative = kappa_aggreement(all_predictions, data_csv)
    # kappas.to_csv(os.path.join(checkpoint_directory, 'kappas.csv'))
    # kappas_positive.to_csv(os.path.join(checkpoint_directory, 'kappas_positive.csv'))
    # kappas_negative.to_csv(os.path.join(checkpoint_directory, 'kappas_negative.csv'))
    all_predictions = pd.merge(all_predictions, data_csv[['episode', 'icustay_id', 'label']], left_index=True,
                                      right_on='episode', how="left")
    all_evaluations = pd.merge(structured_evaluation_predictions, textual_evaluate_predictions,
                               left_index=True, right_index=True, how="left")
    all_evaluations = pd.merge(all_evaluations, data_csv[['episode', 'icustay_id', 'label']], left_index=True,
                                      right_on='episode', how="left")


    # Adding icustay_id to the predictions and representations for merging bellow
    structured_predictions = pd.merge(structured_predictions, data_csv[['episode', 'icustay_id', 'label']], left_index=True,
                                      right_on='episode', how="left")
    structured_representations = pd.merge(structured_representations, data_csv[['episode', 'icustay_id', 'label']], left_index=True,
                                      right_on=['episode'], how="left")
    textual_predictions = pd.merge(textual_predictions, data_csv[['episode', 'icustay_id', 'label']], left_index=True,
                                      right_on=['episode'], how="left")
    textual_representations = pd.merge(textual_representations, data_csv[['episode', 'icustay_id', 'label']], left_index=True,
                                      right_on=['episode'], how="left")
    structured_evaluation_predictions = pd.merge(structured_evaluation_predictions, data_csv[['episode', 'icustay_id', 'label']], left_index=True,
                                      right_on='episode', how="left")
    structured_evaluation_representations = pd.merge(structured_evaluation_representations, data_csv[['episode', 'icustay_id', 'label']], left_index=True,
                                      right_on=['episode'], how="left")
    textual_evaluate_predictions = pd.merge(textual_evaluate_predictions, data_csv[['episode', 'icustay_id', 'label']], left_index=True,
                                      right_on=['episode'], how="left")
    textual_evaluate_representations = pd.merge(textual_evaluate_representations, data_csv[['episode', 'icustay_id', 'label']], left_index=True,
                                      right_on=['episode'], how="left")


    # Get patients non-temporal structured data
    dataset_patients = pandas.read_csv('/home/mattyws/Documents/mimic/sepsis3-df-no-exclusions.csv')
    dataset_patients[['age', 'is_male', 'height', 'weight']] = dataset_patients[['age', 'is_male', 'height', 'weight']].apply(zscore)
    dataset_patients[['age', 'is_male', 'height', 'weight']] = dataset_patients[['age', 'is_male', 'height', 'weight']].fillna(0)
    meta_data_extra = dataset_patients[['icustay_id', 'age', 'is_male', 'height', 'weight']]

    # Testing only structured non-temporal data
    metadata_dataset = pandas.merge(meta_data_extra, data_csv[['episode', 'icustay_id', 'label']], left_on="icustay_id", right_on="icustay_id")
    training_metadata_dataset = metadata_dataset[metadata_dataset['episode'].isin(X)]
    training_metadata_classes = training_metadata_dataset['label']
    training_metadata_dataset = training_metadata_dataset[['age', 'is_male', 'height', 'weight']].values
    meta_adapters = train_meta_model_on_data(training_metadata_dataset, training_metadata_classes, parameters)

    evaluation_metadata_dataset = metadata_dataset[metadata_dataset['episode'].isin(X_val)]
    results, result_predictions = test_meta_model_on_data(meta_adapters, evaluation_metadata_dataset,
                                                          ['age', 'is_male', 'height', 'weight'], 'metadata')
    result_predictions = pandas.DataFrame(result_predictions)
    result_predictions.to_csv(os.path.join(checkpoint_directory, "metamodel_metadata_predictions.csv"))
    results = pandas.DataFrame(results)
    results.to_csv(os.path.join(checkpoint_directory, 'metadata_results.csv'))

    # Using only structured data
    meta_data_predictions_structured = pandas.merge(structured_predictions, meta_data_extra, left_on="icustay_id",
                                                    right_on="icustay_id", how="left")
    meta_evaluation_predictions_structured = pandas.merge(structured_evaluation_predictions, meta_data_extra, left_on="icustay_id",
                                                    right_on="icustay_id", how="left")
    print(meta_data_predictions_structured)
    print(meta_data_predictions_structured.columns)
    # Adding label direct to the data
    # meta_data_predictions_structured = pandas.merge(meta_data_predictions_structured, data_csv[['episode', 'label']],
    #                                                 left_index=True, right_on="episode", how="left")
    # meta_evaluation_predictions_structured = pandas.merge(meta_evaluation_predictions_structured, data_csv[['episode', 'label']],
    #                                                 left_index=True, right_on="episode", how="left")
    print(meta_data_predictions_structured.columns)
    if 'Unnamed: 0' in meta_data_predictions_structured.columns:
        meta_data_predictions_structured = meta_data_predictions_structured.drop(columns=['Unnamed: 0'])
    if 'Unnamed: 0' in meta_evaluation_predictions_structured.columns:
        meta_evaluation_predictions_structured = meta_evaluation_predictions_structured.drop(columns=['Unnamed: 0'])
    columns = [c for c in meta_data_predictions_structured.columns if 'label' not in c and 'icustay' not in c and 'episode' not in c]
    columns_evaluation = [c for c in meta_evaluation_predictions_structured.columns if 'label' not in c and 'icustay' not in c and 'episode' not in c]
    print(columns)
    print(columns_evaluation)
    print(meta_data_predictions_structured.columns)
    training_values = meta_data_predictions_structured.loc[:, columns]
    training_values = training_values.values
    training_classes = np.asarray(meta_data_predictions_structured['label'].tolist())

    testing_values = meta_evaluation_predictions_structured.loc[:, columns_evaluation]
    testing_values = testing_values.values
    testing_classes = np.asarray(meta_evaluation_predictions_structured['label'].tolist())

    meta_adapters = train_meta_model_on_data(training_values, training_classes, parameters)
    # results, result_predictions = test_meta_model_on_data(meta_adapters, testing_values, testing_classes, parameters, 'struct_pred')
    results, result_predictions = test_meta_model_on_data(meta_adapters, meta_evaluation_predictions_structured, columns_evaluation, 'struct_pred')
    ensemble_results.extend(results)
    ensemble_predictions.extend(result_predictions)
    # Using both types of data

    print(all_predictions.columns)
    print(all_evaluations.columns)
    print(meta_data_extra.columns)

    # all_predictions = all_predictions.fillna(-1)
    # all_evaluations = all_evaluations.fillna(-1)
    meta_data_predictions = pandas.merge(all_predictions, meta_data_extra, left_on="icustay_id",
                                                    right_on="icustay_id", how="left")
    meta_data_evaluation = pandas.merge(all_evaluations, meta_data_extra, left_on="icustay_id",
                                                    right_on="icustay_id", how="left")
    print(meta_data_predictions)
    # meta_data_predictions = pandas.merge(meta_data_predictions, data_csv[['episode', 'label']],
    #                                                 left_index=True, right_on="episode", how="left")
    # meta_data_evaluation = pandas.merge(meta_data_evaluation, data_csv[['episode', 'label']],
    #                                                 left_index=True, right_on="episode", how="left")
    print(meta_data_predictions.columns)
    if 'Unnamed: 0' in meta_data_predictions.columns:
        meta_data_predictions = meta_data_predictions.drop(columns=['Unnamed: 0'])
    if 'Unnamed: 0' in meta_data_evaluation.columns:
        meta_data_evaluation = meta_data_evaluation.drop(columns=['Unnamed: 0'])
    columns = [c for c in meta_data_predictions.columns if 'label' not in c and 'icustay' not in c and 'episode' not in c]
    columns_evaluation = [c for c in meta_data_evaluation.columns if 'label' not in c and 'icustay' not in c and 'episode' not in c]
    training_values = meta_data_predictions.loc[:, columns]
    training_values = training_values.values
    training_classes = np.asarray(meta_data_predictions['label'].tolist())

    testing_values = meta_data_evaluation.loc[:, columns_evaluation]
    testing_values = testing_values.values
    testing_classes = np.asarray(meta_data_evaluation['label'].tolist())

    meta_adapters = train_meta_model_on_data(training_values, training_classes, parameters)
    # results, result_predictions = test_meta_model_on_data(meta_adapters, testing_values, testing_classes, parameters, 'both_pred')
    results, result_predictions = test_meta_model_on_data(meta_adapters, meta_data_evaluation, columns_evaluation, 'both_pred')
    ensemble_results.extend(results)
    ensemble_predictions.extend(result_predictions)

    # # REPRESENTATIONS
    #
    # # structured_representations = structured_representations.set_index(pd.to_numeric(structured_representations.index))
    # # print(structured_representations)
    # meta_data_representations_structured = pandas.merge(structured_representations, meta_data_extra, left_on="icustay_id",
    #                                                 right_on="icustay_id", how="left")
    # meta_data_representations_structured.index = meta_data_representations_structured.index.astype(str)
    # meta_evaluation_representations_structured = pandas.merge(structured_evaluation_representations, meta_data_extra,
    #                                                           left_on="icustay_id", right_on="icustay_id", how="left")
    # meta_evaluation_representations_structured.index = meta_evaluation_representations_structured.index.astype(str)
    #
    # print(meta_data_representations_structured)
    # print(meta_data_representations_structured.columns)
    # if 'Unnamed: 0' in meta_data_representations_structured.columns:
    #     meta_data_representations_structured = meta_data_representations_structured.drop(columns=['Unnamed: 0'])
    # if 'Unnamed: 0' in meta_evaluation_representations_structured.columns:
    #     meta_evaluation_representations_structured = meta_evaluation_representations_structured.drop(columns=['Unnamed: 0'])
    # columns = [c for c in meta_data_representations_structured.columns if 'label' not in c and 'icustay' not in c and 'episode' not in c]
    # columns_evaluation = [c for c in meta_evaluation_representations_structured.columns if 'label' not in c and 'icustay' not in c and 'episode' not in c]
    # print(columns)
    # print(meta_data_representations_structured.columns)
    # training_values = meta_data_representations_structured.loc[:, columns]
    # training_values = training_values.values
    # training_classes = np.asarray(meta_data_representations_structured['label'].tolist())
    #
    # testing_values = meta_evaluation_representations_structured.loc[:, columns_evaluation]
    # testing_values = testing_values.values
    # testing_classes = np.asarray(meta_evaluation_representations_structured['label'].tolist())
    #
    # meta_adapters = train_meta_model_on_data(training_values, training_classes, parameters)
    # results = test_meta_model_on_data(meta_adapters, testing_values, testing_classes, parameters, 'struct_repr')
    # ensemble_results.extend(results)
    #
    # textual_columns = [c for c in textual_representations.columns if 't_' in c]
    #
    # all_representations = pd.merge(structured_representations, textual_representations[textual_columns],
    #                                left_index=True, right_index=True, how="left")
    # all_evaluations = pd.merge(structured_evaluation_representations, textual_evaluate_representations[textual_columns],
    #                                left_index=True, right_index=True, how="left")
    # meta_data_representations = pandas.merge(all_representations, meta_data_extra, left_on="icustay_id",
    #                                                 right_on="icustay_id", how="left")
    # meta_evaluation_representations = pandas.merge(all_evaluations, meta_data_extra, left_on="icustay_id",
    #                                                 right_on="icustay_id", how="left")
    # print(meta_data_representations.columns)
    # if 'Unnamed: 0' in meta_data_representations.columns:
    #     meta_data_representations = meta_data_representations.drop(columns=['Unnamed: 0'])
    # if 'Unnamed: 0' in meta_evaluation_representations.columns:
    #     meta_evaluation_representations = meta_evaluation_representations.drop(columns=['Unnamed: 0'])
    # columns = [c for c in meta_data_representations.columns if 'label' not in c and 'icustay' not in c and 'episode' not in c]
    # columns_evaluation = [c for c in meta_evaluation_representations.columns if 'label' not in c and 'icustay' not in c and 'episode' not in c]
    # print(columns)
    # print(meta_data_representations.columns)
    # training_values = meta_data_representations.loc[:, columns]
    # training_values = training_values.values
    # training_classes = np.asarray(meta_data_representations['label'].tolist())
    #
    # testing_values = meta_evaluation_representations.loc[:, columns_evaluation]
    # testing_values = testing_values.values
    # texting_classes = np.asarray(meta_evaluation_representations['label'].tolist())
    #
    # # TODO: Get predictions
    # meta_adapters = train_meta_model_on_data(training_values, training_classes, parameters)
    # results = test_meta_model_on_data(meta_adapters, testing_values, testing_classes, parameters, 'both_repr')
    # ensemble_results.extend(results)

    ensemble_results = pandas.DataFrame(ensemble_results)
    ensemble_results.to_csv(checkpoint_directory + parameters['ensemble_results_filename'])
    all_metrics.to_csv(checkpoint_directory + parameters['metrics_filename'])
    ensemble_predictions = pandas.DataFrame(ensemble_predictions)
    ensemble_predictions.to_csv(os.path.join(checkpoint_directory, 'metamodels_predictions.csv'))