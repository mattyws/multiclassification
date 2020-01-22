parameters = {
    "training_directory_path" : "../mimic/ensemble_training/",
    "dataset_csv_file_path": "../mimic/new_dataset_patients.csv",
    "use_structured_data" : True,
    "use_textual_data": False,

    "normalization_value_counts_dir" : "value_counts/",


    "word2vec_representation_data_path": "transformed_textual_representation/",
    "word2vec_padded_representation_files_path": "padded_textual_representation/",

    "structured_data_path" : "../mimic/structured_data/",
    "textual_data_path" : "../mimic/textual_data/",
    "training_dir_path": "../mimic/ensemble_training/",
    "notes_word2vec_path" : "../mimic/textual_word2vec_preprocessed/",
    "normalized_structured_data_path" : "normalized_data_{}/",

    "checkpoint" : "checkpoint_only_structured/",
    "ensemble_models_path": "ensemble_models_fold_{}/",
    "structured_ensemble_models_name_prefix" : "structured_bagging_level_zero_{}.model",

    "training_config_file_name" : "config.pkl",
    'word2vec_model_file_name': "word2vec.model",
    'meta_model_file_name': 'meta_model_{}.pkl',
    "normalization_values_file_name": "normalization_values_{}.pkl",
    "training_events_sizes_file" : "training_sizes_{}.pkl",
    "training_events_sizes_labels_file" : "training_sizes_labels_{}.pkl",
    "testing_events_sizes_file" : "testing_sizes_{}.pkl",
    "testing_events_sizes_labels_file" : "testing_sizes_labels_{}.pkl",

    "embedding_size" : 150,
    "min_count" : 1,
    "workers" : 4,
    "window" : 3,
    "iterations" : 30,

    "structured_data_length" : 12,
    "structured_output_units": [
        64
    ],
    "strctured_output_neurons": 1,
    "structured_loss": "binary_crossentropy",
    "structured_optimizer":"adam",
    "structured_layers_activations": [
        "relu"
    ],
    "structured_network_activation" : "sigmoid",
    "structured_gru": False,
    "structured_tcn": True,
    "structured_use_dropout": True,
    "structured_dropout": 0.5,
    "structured_training_epochs": 40,
    "structured_batch_size": 50,
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
    ]

    "modelCheckpointPath": "../mimic/ensemble_training/checkpoint/",
    "modelConfigPath": "../mimic/ensemble_training/checkpoint/config.json",
    "trainingDataPath" :  "../mimic/ensemble_training/dataTraining/",
    "testingDataPath" : "../mimic/ensemble_training/dataTest/",
    "datasetFilesFileName": "../mimic/ensemble_training/datasetFiles.pkl",
    "datasetLabelsFileName": "../mimic/ensemble_training/datasetLabels.pkl",
    "trainingGeneratorPath": "../mimic/ensemble_training/checkpoint/dataTrainGenerator.pkl",
    "testingGeneratorPath": "../mimic/ensemble_training/checkpoint/dataTestGenerator.pkl",
    "resultFilePath": "../mimic/ensemble_training/checkpoint/result.csv",
    "temporary_data_path" : "../mimic/ensemble_training/data_tmp_{}/",
    "normalization_data_path": "../mimic/ensemble_training/normalization_values_{}.pkl",

}
