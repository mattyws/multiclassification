parameters = {
    "dataset_csv_file_path": "../mimic/new_dataset_patients.csv",
    "use_structured_data" : True,
    "use_textual_data": False,

    "structured_data_path" : "../mimic/structured_data/",
    "textual_data_path" : "../mimic/textual_data/",
    "training_dir_path": "../mimic/ensemble_training/",

    "checkpoint" : "checkpoint_only_structured/",
    "ensemble_models_path": "ensemble_models_fold_{}/",
    "structured_ensemble_models_name_prefix" : "structured_bagging_level_zero_{}.model",

    "training_config_file_name" : "config.pkl",



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
    "normalization_value_counts_path" : "../mimic/ensemble_training/value_counts/",
    "training_events_sizes_file" : "../mimic/ensemble_training/training_sizes_{}.pkl",
    "training_events_sizes_labels_file" : "../mimic/ensemble_training/training_sizes_labels_{}.pkl",
    "testing_events_sizes_file" : "../mimic/ensemble_training/testing_sizes_{}.pkl",
    "testing_events_sizes_labels_file" : "../mimic/ensemble_training/testing_sizes_labels_{}.pkl",
    "dataLength" : 12,
    "outputUnits": [
        64
    ],
    "numOutputNeurons": 1,
    "loss": "binary_crossentropy",
    "optimizer":"adam",
    "layersActivations": [
        "relu"
    ],
    "networkActivation" : "sigmoid",
    "gru": True,
    "useDropout": True,
    "dropout": 0.5,
    "trainingEpochs": 40,
    "batchSize": 50
}
