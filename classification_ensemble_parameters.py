parameters = {
    "datasetCsvFilePath": "../mimic/dataset_patients.csv",
    "modelCheckpointPath": "../mimic/articles_features_training/checkpoint_1/",
    "modelConfigPath": "../mimic/articles_features_training/checkpoint_1/config.json",
    "dataPath" : "../mimic/sepsis_articles_bucket/",
    "trainingDataPath" :  "../mimic/articles_features_training/dataTraining/",
    "testingDataPath" : "../mimic/articles_features_training/dataTest/",
    "datasetFilesFileName": "../mimic/articles_features_training/datasetFiles.pkl",
    "datasetLabelsFileName": "../mimic/articles_features_training/datasetLabels.pkl",
    "trainingGeneratorPath": "../mimic/articles_features_training/checkpoint_1/dataTrainGenerator.pkl",
    "testingGeneratorPath": "../mimic/articles_features_training/checkpoint_1/dataTestGenerator.pkl",
    "resultFilePath": "../mimic/articles_features_training/checkpoint_1/result.csv",
    "temporary_data_path" : "../mimic/articles_features_training/data_tmp_{}/",
    "normalization_data_path": "../mimic/articles_features_training/normalization_values_{}.pkl",
    "normalization_value_counts_path" : "../mimic/articles_features_training/value_counts/",
    "training_events_sizes_file" : "../mimic/articles_features_training/training_sizes_{}.pkl",
    "training_events_sizes_labels_file" : "../mimic/articles_features_training/training_sizes_labels_{}.pkl",
    "testing_events_sizes_file" : "../mimic/articles_features_training/testing_sizes_{}.pkl",
    "testing_events_sizes_labels_file" : "../mimic/articles_features_training/testing_sizes_labels_{}.pkl",
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
