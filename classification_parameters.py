from keras.layers import LeakyReLU

parameters = {
    "datasetCsvFilePath": "../mimic/new_dataset_patients.csv",
    "modelCheckpointPath": "../mimic/new_filtered_training/checkpoint_gru/",
    "modelConfigPath": "../mimic/new_filtered_training/checkpoint_gru/config.json",
    "dataPath" : "../mimic/structured_data/",
    "trainingDataPath" :  "../mimic/new_filtered_training/dataTraining/",
    "testingDataPath" : "../mimic/new_filtered_training/dataTest/",
    "datasetFilesFileName": "../mimic/new_filtered_training/datasetFiles.pkl",
    "datasetLabelsFileName": "../mimic/new_filtered_training/datasetLabels.pkl",
    "trainingGeneratorPath": "../mimic/new_filtered_training/checkpoint_gru/dataTrainGenerator.pkl",
    "testingGeneratorPath": "../mimic/new_filtered_training/checkpoint_gru/dataTestGenerator.pkl",
    "resultFilePath": "../mimic/new_filtered_training/checkpoint_gru/result.csv",
    "temporary_data_path" : "../mimic/new_filtered_training/data_tmp_{}/",
    "normalization_data_path": "../mimic/new_filtered_training/normalization_values_{}.pkl",
    "normalization_value_counts_path" : "../mimic/new_filtered_training/value_counts/",
    "training_events_sizes_file" : "../mimic/new_filtered_training/training_sizes_{}.pkl",
    "training_events_sizes_labels_file" : "../mimic/new_filtered_training/training_sizes_labels_{}.pkl",
    "testing_events_sizes_file" : "../mimic/new_filtered_training/testing_sizes_{}.pkl",
    "testing_events_sizes_labels_file" : "../mimic/new_filtered_training/testing_sizes_labels_{}.pkl",
    "dataLength" : 12,
    "outputUnits": [
        64
    ],
    "numOutputNeurons": 1,
    "loss": "binary_crossentropy",
    "optimizer":"adam",
    "layersActivations": [
        LeakyReLU()
    ],
    "networkActivation" : "sigmoid",
    "gru": True,
    "tcn": True,
    "useDropout": True,
    "dropout": 0.5,
    "trainingEpochs": 2,
    "batchSize": 50
}
