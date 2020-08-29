from tensorflow.keras.layers import Activation
from tensorflow.keras import activations
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.optimizers import Adam
import os

path_parameters = {
    "multiclassification_base_path": os.path.expanduser("~/Documents/mimic/multiclassification/"),
    "training_directory_path" : "0-Training/",

    "mortality_directory" : "mortality/",
    "mortality_dataset_csv" : "mortality.csv",

    "decompensation_directory": "decompensation/",
    "decompensation_dataset_csv" : "decompensation.csv"
}

timeseries_execution_saving_parameters = {
    "execution_saving_path" : "timeseries_w_opt/",
    "training_checkpoint": "checkpoint_unweighted/",
    "execution_parameters_filename": "training_parameters.pkl",

    "normalization_value_counts_directory" : "value_counts/",
    "fold_normalization_values_filename": "normalization_values_{}.pkl",
    "fold_normalization_temporary_data_directory" : "data_tmp_{}/",
    'tunning_directory' : 'tunning/',

    "training_events_sizes_filename" : "training_sizes_{}.pkl",
    "training_events_sizes_labels_filename" : "training_sizes_labels_{}.pkl",
    "testing_events_sizes_filename" : "testing_sizes_{}.pkl",
    "testing_events_sizes_labels_filename" : "testing_sizes_labels_{}.pkl",
    "training_samples_filename" : "training_samples.pkl",
    "training_classes_filename" : "training_classes.pkl",
    "optimization_samples_filename" : "optimization_samples.pkl",
    "optimization_classes_filename" : "optimization_classes.pkl",

    "result_filename" : "result.csv",
    "trained_model_filename" : "trained_{}.model"
}

timeseries_model_parameters = {
    "outputUnits": [
        16,
        16
    ],
    "numOutputNeurons": 1,
    "loss": "binary_crossentropy",
    "optimizer": Adam(
        learning_rate=0.01,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-07,
        amsgrad=True
    ),
    "layersActivations": [
        LeakyReLU(),
        LeakyReLU()
    ],
    "networkActivation" : "sigmoid",
    "gru": False,
    "tcn": False,
    "useDropout": True,
    "dropout": 0.3,
    "trainingEpochs": 28,
    "batchSize": 8,

    # Convolution only parameters
    "kernel_sizes": [
        3
    ],
    "pooling": [
        False
    ],
    "dilations": [
        [1, 2, 4]
    ],
    "nb_stacks": [
        1
    ],

    'model_tunning': True,
    'optimization_split_rate': .20,
    'optimization_normalization_values_filename': "opt_normalization_values.pkl",
    'optimization_normalization_temporary_data_directory': "opt_data_tmp/"
}

textual_execution_saving_parameters = {
    "execution_saving_path" : "texts/",
    "training_checkpoint": "checkpoint_all_mean/",
    "execution_parameters_filename": "training_parameters.pkl",

    "bert_directory": os.path.expanduser("~/Documents/mimic/bert/"),
    "tokenization_strategy": "all",
    "sentence_encoding_strategy" : "mean",

    'tunning_directory' : 'tunning/',

    "training_events_sizes_filename" : "training_sizes_{}.pkl",
    "training_events_sizes_labels_filename" : "training_sizes_labels_{}.pkl",
    "testing_events_sizes_filename" : "testing_sizes_{}.pkl",
    "testing_events_sizes_labels_filename" : "testing_sizes_labels_{}.pkl",
    "training_samples_filename" : "training_samples.pkl",
    "training_classes_filename" : "training_classes.pkl",
    "optimization_samples_filename" : "optimization_samples.pkl",
    "optimization_classes_filename" : "optimization_classes.pkl",

    "result_filename" : "result.csv",
    "trained_model_filename" : "trained_{}.model"
}

textual_model_parameters = {
    "outputUnits": [
        16,
        16
    ],
    "numOutputNeurons": 1,
    "loss": "binary_crossentropy",
    "optimizer": Adam(
        learning_rate=0.01,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-07,
        amsgrad=True
    ),
    "layersActivations": [
        LeakyReLU(),
        LeakyReLU()
    ],
    "networkActivation" : "sigmoid",
    "gru": False,
    "tcn": False,
    "useDropout": True,
    "dropout": 0.3,
    "trainingEpochs": 28,
    "batchSize": 8,

    # Convolution only parameters
    "kernel_sizes": [
        3
    ],
    "pooling": [
        False
    ],
    "dilations": [
        [1, 2, 4]
    ],
    "nb_stacks": [
        1
    ],

    'model_tunning': True,
    'optimization_split_rate': .20,
    'optimization_normalization_values_filename': "opt_normalization_values.pkl",
    'optimization_normalization_temporary_data_directory': "opt_data_tmp/"
}

model_tuner_parameters = {
    'layers_min': 1,
    'layers_max' : 2,
    'layers_step': 1,
    'hidden_output_units_min' : 8,
    'hidden_output_units_max': 32,
    'hidden_output_units_steps': 8,
    'losses': ['binary_crossentropy'],
    'hidden_activations': ["leakyrelu"],
    'network_activations': ["sigmoid"],
    'hidden_pooling': [False],
    'kernel_size_min': 3,
    'kernel_size_max': 5,
    'kernel_size_step': 1,
    'min_dilation_size': 3,
    'max_dilation_size': 4,
    'dilation_size_step': 1,
    'min_dilation': 1,
    'max_dilation': 4,
    'dilation_step': 1,
    'stacks_min': 1,
    'stacks_max': 1,
    'stacks_steps': 1,
    'use_dropout': [True, False],
    'dropout_choices': [.2, .3, .5],
    'optimizers': ["adam"]
}

timeseries_tuning_parameters = model_tuner_parameters
textual_tuning_parameters = {
'layers_min': 1,
    'layers_max' : 2,
    'layers_step': 1,
    'hidden_output_units_min' : 8,
    'hidden_output_units_max': 128,
    'hidden_output_units_steps': 16,
    'losses': ['binary_crossentropy'],
    'hidden_activations': ["leakyrelu"],
    'network_activations': ["sigmoid"],
    'hidden_pooling': [False],
    'kernel_size_min': 2,
    'kernel_size_max': 10,
    'kernel_size_step': 2,
    'min_dilation_size': 3,
    'max_dilation_size': 4,
    'dilation_size_step': 1,
    'min_dilation': 1,
    'max_dilation': 4,
    'dilation_step': 1,
    'stacks_min': 1,
    'stacks_max': 1,
    'stacks_steps': 1,
    'use_dropout': [True, False],
    'dropout_choices': [.2, .3, .5],
    'optimizers': ["adam"]
}

ensemble_stacking_parameters = {
    "execution_saving_path" : "mixed_model/",
    "training_checkpoint" : "checkpoint/",

    "use_structured_data" : True,
    "use_textual_data": True,
    "use_class_weight": False,

    "train_test_split_rate": .10,
    "training_samples_filename" : "training_samples.pkl",
    "training_classes_filename" : "training_classes.pkl",
    "testing_samples_filename" : "testing_samples.pkl",
    "testing_classes_filename" : "testing_classes.pkl",

    'model_tunning': True,
    'optimization_split_rate': .15,
    'optimization_normalization_values_filename': "opt_normalization_values.pkl",
    'optimization_normalization_temporary_data_directory': "opt_data_tmp/",
    'tunning_directory' : 'tunning/',
    "optimization_samples_filename" : "optimization_samples.pkl",
    "optimization_classes_filename" : "optimization_classes.pkl",

    "level_zero_result_filename": "level_zero_result.csv",

    "fold_metrics_filename" : "fold_result_{}.csv",
    "metrics_filename" : "result.csv",
    "ensemble_results_filename" : "ensemble_result.csv",


    "fold_structured_predictions_filename" : "fold_structured_prediction_{}.csv",
    "fold_structured_representations_filename": "fold_structured_representation_{}.csv",
    "fold_textual_predictions_filename" : "fold_textual_prediction_{}.csv",
    "fold_textual_representations_filename": "fold_textual_representation_{}.csv",

    "structured_predictions_filename" : "structured_prediction.csv",
    "structured_representations_filename": "structured_representations.csv",
    "textual_predictions_filename" : "textual_prediction.csv",
    "textual_representations_filename": "textual_representations.csv",

    "fold_normalization_values_filename": "normalization_values_{}.pkl",
    "fold_normalization_temporary_data_directory" : "data_tmp_{}/",
    "normalization_value_counts_dir" : "value_counts/",

    "structured_training_events_sizes_filename" : "structured_training_sizes_{}.pkl",
    "structured_training_events_sizes_labels_filename" : "structured_training_sizes_labels_{}.pkl",
    "structured_testing_events_sizes_filename" : "structured_testing_sizes_{}.pkl",
    "structured_testing_events_sizes_labels_filename" : "structured_testing_sizes_labels_{}.pkl",
    "textual_training_events_sizes_filename" : "textual_training_sizes_{}.pkl",
    "textual_training_events_sizes_labels_filename" : "textual_training_sizes_labels_{}.pkl",
    "textual_testing_events_sizes_filename" : "textual_testing_sizes_{}.pkl",
    "textual_testing_events_sizes_labels_filename" : "textual_testing_sizes_labels_{}.pkl",

    "structured_weak_model": "structured_weak_{}_{}.model",
    "textual_weak_model": "textual_weak_{}_{}.model",


    "structured_weak_model_all": "structured_weak_all_{}.model",
    "textual_weak_model_all": "textual_weak_all_{}.model",


    "notes_textual_representation_directory" : "notes_transformed_representation/",
    'textual_representation_model_path': os.path.expanduser("~/Documents/mimic/trained_doc2vec/"),
    'textual_representation_model_filename' : 'doc2vec.model',

    "normalized_structured_data_path" : "normalized_data_{}/",
    "normalization_data_path": "normalization_values_{}.pkl",

    "ensemble_models_path": "ensemble_models_fold_{}/",


    "structured_ensemble_models_name_prefix" : "structured_bagging_level_zero_{}.model",
    "structured_ensemble_samples_name_prefix" : "structured_bagging_level_zero_samples_{}.pkl",
    "textual_ensemble_models_name_prefix" : "textual_bagging_level_zero_{}.model",
    "textual_ensemble_samples_name_prefix" : "textual_bagging_level_zero_samples_{}.pkl",
    "meta_model_file_name": "{}_meta_model_fold_{}.model",

    "training_config_file_name" : "config.pkl",

    "normalization_values_file_name": "normalization_values_{}.pkl",
    "level_zero_structured_result_file_name": "structured_results_{}.csv",
    "level_zero_textual_result_file_name": "textual_results_{}.csv",

    "structured_training_events_sizes_file" : "structured_training_sizes_{}.pkl",
    "structured_training_events_sizes_labels_file" : "structured_training_sizes_labels_{}.pkl",
    "textual_training_events_sizes_file" : "textual_training_sizes_{}.pkl",
    "textual_training_events_sizes_labels_file" : "textual_training_sizes_labels_{}.pkl",

    "structured_testing_events_sizes_file" : "structured_testing_sizes_{}.pkl",
    "structured_testing_events_sizes_labels_file" : "structured_testing_sizes_labels_{}.pkl",
    "textual_testing_events_sizes_file" : "textual_testing_sizes_{}.pkl",
    "textual_testing_events_sizes_labels_file" : "textual_testing_sizes_labels_{}.pkl",

    "structured_output_units": [
        16, 16
    ],
    "structured_output_neurons": 1,
    "structured_loss": "binary_crossentropy",
    "structured_optimizer": Adam(
        learning_rate=0.01,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-07,
        amsgrad=True
    ),
    "structured_layers_activations": [
        LeakyReLU(), LeakyReLU()
    ],
    "structured_network_activation" : "sigmoid",
    "structured_gru": False,
    "structured_tcn": False,
    "structured_use_dropout": True,
    "structured_dropout": 0.3,
    "structured_training_epochs": 30,
    "structured_batch_size": 6,
    # Temporal convolutional network parameters only
    "structured_kernel_sizes": [
        3, 3
    ],
    "structured_pooling": [
        False, False
    ],
    "structured_dilations": [
        [1, 2, 4], [1, 2, 4]
    ],
    "structured_nb_stacks": [
        1, 1
    ],

    "textual_embedding_size" : 300,
    "textual_min_count" : 10,
    "textual_workers" : 4,
    "textual_window" : 3,
    "textual_iterations" : 5,
    "textual_doc2vec_dm" : 0,
    "textual_doc2vec_hs": 0,
    "textual_doc2vec_negative": 5,
    "textual_gru": False,
    "textual_use_dropout" : True,
    "textual_dropout": 0.3,
    "textual_batch_size" : 8,

    "textual_output_units": [
        16, 16
    ],
    "textual_output_neurons": 1,
    "textual_loss": "binary_crossentropy",
    "textual_optimizer": Adam(
        learning_rate=0.01,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-07,
        amsgrad=True
    ),
    "textual_layers_activations": [
        LeakyReLU(), LeakyReLU()
    ],
    "textual_network_activation" : "sigmoid",
    "textual_training_epochs": 28,
# Temporal convolutional network parameters only
    "textual_kernel_sizes": [
        3, 3
    ],
    "textual_pooling": [
        False, False
    ],
    "textual_dilations": [
        [1, 2, 4], [1, 2, 4]
    ],
    "textual_nb_stacks": [
        1, 1
    ],

    'meta_learner_batch_size': 8,
    'meta_learner_output_units': [
        32, 16, 8
    ],
    'meta_learner_num_output_neurons': 1,
    'meta_learner_loss': 'binary_crossentropy',
    'meta_learner_layers_activations': [
        Activation(activations.relu),
        Activation(activations.relu),
        Activation(activations.relu)
    ],
    'meta_learner_network_activation': 'sigmoid',
    'meta_learner_use_dropout': True,
    'meta_learner_dropout': 0.3,
    "meta_learner_optimizer": Adam(
        learning_rate=0.01,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-07,
        amsgrad=True
    ),
    "meta_learner_training_epochs": 60,
    "use_class_prediction": False,

    "structured_model_input_name": "structured",
    "textual_model_input_name": "textual"
}

timeseries_training_parameters = dict()
timeseries_training_parameters.update(path_parameters)
timeseries_training_parameters.update(timeseries_execution_saving_parameters)
timeseries_training_parameters.update(timeseries_model_parameters)

timeseries_textual_training_parameters = dict()
timeseries_textual_training_parameters.update(path_parameters)
timeseries_textual_training_parameters.update(textual_execution_saving_parameters)
timeseries_textual_training_parameters.update(textual_model_parameters)

ensemble_stacking_parameters.update(path_parameters)
