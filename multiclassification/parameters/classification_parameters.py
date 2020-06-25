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
    "execution_saving_path" : "timeseries/",
    "training_checkpoint": "checkpoint_unweighted/",
    "execution_parameters_filename": "training_parameters.pkl",

    "normalization_value_counts_directory" : "value_counts/",
    "fold_normalization_values_filename": "normalization_values_{}.pkl",
    "fold_normalization_temporary_data_directory" : "data_tmp_{}/",

    "training_events_sizes_filename" : "training_sizes_{}.pkl",
    "training_events_sizes_labels_filename" : "training_sizes_labels_{}.pkl",
    "testing_events_sizes_filename" : "testing_sizes_{}.pkl",
    "testing_events_sizes_labels_filename" : "testing_sizes_labels_{}.pkl",

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
    ]
}

ensemble_stacking_parameters = {
    "execution_saving_path" : "ensemble_training_stacking/",
    "training_checkpoint" : "checkpoint2/",

    "use_structured_data" : True,
    "use_textual_data": True,
    "use_class_weight": False,

    "train_test_split_rate": .10,
    "training_samples_filename" : "training_samples.pkl",
    "training_classes_filename" : "training_classes.pkl",
    "testing_samples_filename" : "testing_samples.pkl",
    "testing_classes_filename" : "testing_classes.pkl",

    "level_zero_result_filename": "level_zero_result.csv",

    "fold_metrics_filename" : "fold_result_{}.csv",
    "metrics_filename" : "result.csv",
    "ensemble_results_file_name" : "ensemble_result.csv",


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

    "structured_weak_model_all": "structured_weak_all_{}.model",
    "textual_weak_model_all": "textual_weak_all_{}.model",


    "notes_textual_representation_directory" : "notes_transformed_representation/",
    'textual_representation_model_path': os.path.expanduser("~/Documents/mimic/trained_doc2vec/100/doc2vec.model"),

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

    "textual_embedding_size" : 100,
    "textual_min_count" : 1,
    "textual_workers" : 4,
    "textual_window" : 3,
    "textual_use_dropout" : True,
    "textual_dropout": 0.3,
    "textual_iterations" : 30,
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

    'meta_learner_batch_size': 16,
    'meta_learner_output_units': [
        8
    ],
    'meta_learner_num_output_neurons': 1,
    'meta_learner_loss': 'binary_crossentropy',
    'meta_learner_layers_activations': [
        LeakyReLU()
    ],
    'meta_learner_network_activation': 'sigmoid',
    'meta_learner_use_dropout': True,
    'meta_learner_dropout': 0.3,
    "meta_learner_optimizer":"adam",
    "meta_learner_training_epochs": 40,
    "use_class_prediction": False,


    # For clustering only
    "encoder_model_filename": "encoder_vae.model",
    "decoder_model_filename": "decoder_vae.model",
    "distance_matrix_filename": "distances_{}.pkl",
    "vae_model_filename": "vae.model",
    "encoded_data_path": "encoder_encoded_data_{}/",
    "clustering_ensemble_models_path": "{}_clusters_ensemble_models_fold_{}/",
    "cluster_model_filename": "{}_cluster_{}.model",
    "data_samples_filename": "{}_data_samples_{}.pkl",
    "classes_samples_filename": "{}_classes_samples_{}.pkl",
    "autoencoder_batch_size": 50,
    "encoded_dim": 64,
    "decoder_latent_dim": 72

}


timeseries_training_parameters = dict()
timeseries_training_parameters.update(path_parameters)
timeseries_training_parameters.update(timeseries_execution_saving_parameters)
timeseries_training_parameters.update(timeseries_model_parameters)

ensemble_stacking_parameters.update(path_parameters)
