from tensorflow.keras.layers import LeakyReLU, ReLU

parameters = {
    "training_directory_path" : "../mimic/ensenble_training_stacking/",
    "dataset_csv_file_path": "../mimic/new_dataset_patients.csv",
    "ensemble_training_method" : "bagging",
    "use_structured_data" : True,
    "use_textual_data": True,

    "fold_predictions_file_csv" : "fold_prediction_{}.csv",
    "fold_metrics_file_csv" : "fold_result_{}.csv",
    "predictions_file_csv" : "prediction.csv",
    "metrics_file_csv" : "result.csv",

    "normalization_value_counts_dir" : "value_counts/",
    "meta_representation_path": "meta_representation_fold_{}/",


    "textual_representation_representation_data_path": "transformed_textual_representation/",
    "textual_representation_padded_representation_files_path": "padded_textual_representation/",

    "structured_data_path" : "../mimic/structured_data/",
    "textual_data_path" : "../mimic/textual_anonymized_data/",
    "training_dir_path": "../mimic/ensemble_training/",
    "notes_textual_representation_path" : "../mimic/trained_doc2vec/50/transformed_representation/",
    "normalized_structured_data_path" : "normalized_data_{}/",
    "normalization_data_path": "normalization_values_{}.pkl",

    "checkpoint" : "checkpoint/",
    "ensemble_models_path": "ensemble_models_fold_{}/",
    "structured_ensemble_models_name_prefix" : "structured_bagging_level_zero_{}.model",
    "structured_ensemble_samples_name_prefix" : "structured_bagging_level_zero_samples_{}.pkl",
    "textual_ensemble_models_name_prefix" : "textual_bagging_level_zero_{}.model",
    "textual_ensemble_samples_name_prefix" : "textual_bagging_level_zero_samples_{}.pkl",
    "meta_model_file_name": "{}_meta_model_fold_{}.model",

    "training_config_file_name" : "config.pkl",
    'textual_representation_model_path': "../mimic/trained_doc2vec/50/doc2vec.model",
    "normalization_values_file_name": "normalization_values_{}.pkl",
    "results_file_name": "result.csv",
    "level_zero_structured_result_file_name": "structured_results_{}.csv",
    "level_zero_textual_result_file_name": "textual_results_{}.csv",
    "level_zero_result_file_name": "level_zero_result.csv",

    "structured_training_events_sizes_file" : "structured_testing_sizes_{}.pkl",
    "structured_training_events_sizes_labels_file" : "structured_testing_sizes_labels_{}.pkl",
    "textual_training_events_sizes_file" : "textual_testing_sizes_{}.pkl",
    "textual_training_events_sizes_labels_file" : "textual_testing_sizes_labels_{}.pkl",

    "structured_testing_events_sizes_file" : "structured_testing_sizes_{}.pkl",
    "structured_testing_events_sizes_labels_file" : "structured_testing_sizes_labels_{}.pkl",
    "textual_testing_events_sizes_file" : "textual_testing_sizes_{}.pkl",
    "textual_testing_events_sizes_labels_file" : "textual_testing_sizes_labels_{}.pkl",

    "n_estimators": 3,
    "dataset_split_rate": 1.4,

    "structured_output_units": [
        32
    ],
    "structured_output_neurons": 1,
    "structured_loss": "binary_crossentropy",
    "structured_optimizer":"adam",
    "structured_layers_activations": [
        LeakyReLU()
    ],
    "structured_network_activation" : "sigmoid",
    "structured_gru": False,
    "structured_tcn": True,
    "structured_use_dropout": True,
    "structured_dropout": 0.5,
    "structured_training_epochs": 2,
    "structured_n_estimators": 15,
    "structured_batch_size": 16,
    # Temporal convolutional network parameters only
    "structured_kernel_sizes": [
        3
    ],
    "structured_pooling": [
        False
    ],
    "structured_dilations": [
        [1, 2, 4]
    ],
    "structured_nb_stacks": [
        1
    ],

    "textual_embedding_size" : 50,
    "textual_min_count" : 1,
    "textual_workers" : 4,
    "textual_window" : 3,
    "textual_use_dropout" : True,
    "textual_dropout": 0.3,
    "textual_iterations" : 30,
    "textual_batch_size" : 16,

    "textual_output_units": [
        16
    ],
    "textual_output_neurons": 1,
    "textual_loss": "binary_crossentropy",
    "textual_optimizer":"adam",
    "textual_layers_activations": [
        LeakyReLU()
    ],
    "textual_network_activation" : "sigmoid",
    "textual_training_epochs": 2,
# Temporal convolutional network parameters only
    "textual_kernel_sizes": [
        3
    ],
    "textual_pooling": [
        False
    ],
    "textual_dilations": [
        [1, 2, 4]
    ],
    "textual_nb_stacks": [
        1
    ],

    'meta_learner_batch_size': 16,
    'meta_learner_output_units': [
        32
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
    "meta_learner_training_epochs": 2,
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
