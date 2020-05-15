from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.optimizers import Adam

parameters = {
    "datasetCsvFilePath": "../mimic/new_dataset_patients.csv",
    "modelCheckpointPath": "../mimic/word2vec_raw_training/checkpoint_50/",
    "word2vecModelFileName": "../mimic/trained_word2vec/50/word2vec.model",
    "word2vec_representation_files_path": "../mimic/trained_word2vec/50/transformed_representation_ctakes/",
    "word2vec_padded_representation_files_path": "../mimic/word2vec_raw_training/padded_representation/",
    "embedding_size" : 50,
    "min_count" : 1,
    "workers" : 4,
    "window" : 4,
    "iterations" : 90,

    "modelConfigPath": "../mimic/word2vec_raw_training/checkpoint_50/config.json",
    "savedModelPath": "../mimic/word2vec_raw_training/checkpoint_50/trained_model_{}.model",
    "dataPath" : "../mimic/textual_normalized_preprocessed/",
    "notes_word2vec_path" : "../mimic/sepsis_noteevents_preprocessed/",
    "notes_ctakes_path" : "../mimic/sepsis_noteevents_processed_ctakes/",
    "trainingDataPath" :  "../mimic/word2vec_raw_training/dataTraining/",
    "testingDataPath" : "../mimic/word2vec_raw_training/dataTest/",
    "datasetFilesFileName": "../mimic/word2vec_raw_training/datasetFiles.pkl",
    "datasetLabelsFileName": "../mimic/word2vec_raw_training/datasetLabels.pkl",
    "trainingGeneratorPath": "../mimic/word2vec_raw_training/checkpoint_50/dataTrainGenerator.pkl",
    "testingGeneratorPath": "../mimic/word2vec_raw_training/checkpoint_50/dataTestGenerator.pkl",
    "resultFilePath": "../mimic/word2vec_raw_training/checkpoint_50/result.csv",
    "temporary_data_path" : "../mimic/word2vec_raw_training/data_tmp_{}/",
    "normalization_data_path": "../mimic/word2vec_raw_training/normalization_values_{}.pkl",
    "normalization_value_counts_path" : "../mimic/word2vec_raw_training/value_counts/",
    "training_events_sizes_file" : "../mimic/word2vec_raw_training/training_sizes_{}.pkl",
    "training_events_sizes_labels_file" : "../mimic/word2vec_raw_training/training_sizes_labels_{}.pkl",
    "testing_events_sizes_file" : "../mimic/word2vec_raw_training/testing_sizes_{}.pkl",
    "testing_events_sizes_labels_file" : "../mimic/word2vec_raw_training/testing_sizes_labels_{}.pkl",
    "dataLength" : 12,
    "outputUnits": [
        16
    ],
    "numOutputNeurons": 1,
    "loss": "binary_crossentropy",
    "optimizer": Adam(
        learning_rate=0.001,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-07,
        amsgrad=True
    ),
    "layersActivations": [
        LeakyReLU()
    ],
    "networkActivation" : "sigmoid",
    "gru": True,
    "useDropout": True,
    "dropout": 0.2,
    "trainingEpochs": 50,
    "batchSize": 16,
    "kernel_sizes": [
        3
    ],
    "pooling": [
        True
    ],
    "dilations": [
        [1, 2, 4]
    ],
    "nb_stacks": [
        1
    ]
}
