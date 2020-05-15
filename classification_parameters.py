from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.optimizers import Adam

parameters = {
    "datasetCsvFilePath": "../mimic/new_dataset_patients.csv",
    "modelCheckpointPath": "../mimic/new_filtered_training/checkpoint_tcn_1/",
    "modelConfigPath": "../mimic/new_filtered_training/checkpoint_tcn_1/config.json",
    "dataPath" : "../mimic/structured_data/",
    "trainingDataPath" :  "../mimic/new_filtered_training/dataTraining/",
    "testingDataPath" : "../mimic/new_filtered_training/dataTest/",
    "datasetFilesFileName": "../mimic/new_filtered_training/datasetFiles.pkl",
    "datasetLabelsFileName": "../mimic/new_filtered_training/datasetLabels.pkl",
    "trainingGeneratorPath": "../mimic/new_filtered_training/checkpoint_tcn_1/dataTrainGenerator.pkl",
    "testingGeneratorPath": "../mimic/new_filtered_training/checkpoint_tcn_1/dataTestGenerator.pkl",
    "resultFilePath": "../mimic/new_filtered_training/checkpoint_tcn_1/result.csv",
    "temporary_data_path" : "../mimic/new_filtered_training/data_tmp_{}/",
    "normalization_data_path": "../mimic/new_filtered_training/normalization_values_{}.pkl",
    "normalization_value_counts_path" : "../mimic/new_filtered_training/value_counts/",
    "training_events_sizes_file" : "../mimic/new_filtered_training/training_sizes_{}.pkl",
    "training_events_sizes_labels_file" : "../mimic/new_filtered_training/training_sizes_labels_{}.pkl",
    "testing_events_sizes_file" : "../mimic/new_filtered_training/testing_sizes_{}.pkl",
    "testing_events_sizes_labels_file" : "../mimic/new_filtered_training/testing_sizes_labels_{}.pkl",
    "dataLength" : 12,
    "outputUnits": [
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
        LeakyReLU()
    ],
    "networkActivation" : "sigmoid",
    "gru": False,
    "tcn": True,
    "useDropout": True,
    "dropout": 0.3,
    "trainingEpochs": 40,
    "batchSize": 16,

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
