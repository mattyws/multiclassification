from keras.layers import LeakyReLU

parameters = {
    "datasetCsvFilePath": "../mimic/new_dataset_patients.csv",
    "modelCheckpointPath": "../mimic/new_filtered_training/checkpoint_tcn/",
    "modelConfigPath": "../mimic/new_filtered_training/checkpoint_tcn/config.json",
    "dataPath" : "../mimic/structured_data/",
    "trainingDataPath" :  "../mimic/new_filtered_training/dataTraining/",
    "testingDataPath" : "../mimic/new_filtered_training/dataTest/",
    "datasetFilesFileName": "../mimic/new_filtered_training/datasetFiles.pkl",
    "datasetLabelsFileName": "../mimic/new_filtered_training/datasetLabels.pkl",
    "trainingGeneratorPath": "../mimic/new_filtered_training/checkpoint_tcn/dataTrainGenerator.pkl",
    "testingGeneratorPath": "../mimic/new_filtered_training/checkpoint_tcn/dataTestGenerator.pkl",
    "resultFilePath": "../mimic/new_filtered_training/checkpoint_tcn/result.csv",
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
    "gru": False,
    "tcn": True,
    "useDropout": True,
    "dropout": 0.3,
    "trainingEpochs": 40,
    "batchSize": 50,

    # Convolution only parameters
    "kernel_sizes": [
        3
    ],
    "pooling": [
        False
    ],
    "dilations": [
        [1, 2, 4]
    ]
}
