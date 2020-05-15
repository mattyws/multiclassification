from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.optimizers import Adam

parameters = {
    "datasetCsvFilePath": "../mimic/new_dataset_patients.csv",
    "modelCheckpointPath": "../mimic/doc2vec_raw_training/checkpoint_50/",
    "word2vecModelFileName": "../mimic/trained_doc2vec/50/doc2vec.model",
    "word2vec_representation_files_path": "../mimic/trained_doc2vec/50/transformed_representation/",
    "word2vec_padded_representation_files_path": "../mimic/doc2vec_raw_training/padded_representation/",
    "embedding_size" : 50,
    "min_count" : 1,
    "workers" : 4,
    "window" : 4,
    "iterations" : 90,

    "modelConfigPath": "../mimic/doc2vec_raw_training/checkpoint_50/config.json",
    "savedModelPath": "../mimic/doc2vec_raw_training/checkpoint_50/trained_model_{}.model",
    "dataPath" : "../mimic/textual_normalized_preprocessed/",
    "notes_word2vec_path" : "../mimic/sepsis_noteevents_preprocessed/",
    "trainingDataPath" :  "../mimic/doc2vec_raw_training/dataTraining/",
    "testingDataPath" : "../mimic/doc2vec_raw_training/dataTest/",
    "datasetFilesFileName": "../mimic/doc2vec_raw_training/datasetFiles.pkl",
    "datasetLabelsFileName": "../mimic/doc2vec_raw_training/datasetLabels.pkl",
    "trainingGeneratorPath": "../mimic/doc2vec_raw_training/checkpoint_50/dataTrainGenerator.pkl",
    "testingGeneratorPath": "../mimic/doc2vec_raw_training/checkpoint_50/dataTestGenerator.pkl",
    "resultFilePath": "../mimic/doc2vec_raw_training/checkpoint_50/result.csv",
    "temporary_data_path" : "../mimic/doc2vec_raw_training/data_tmp_{}/",
    "normalization_data_path": "../mimic/doc2vec_raw_training/normalization_values_{}.pkl",
    "normalization_value_counts_path" : "../mimic/doc2vec_raw_training/value_counts/",
    "training_events_sizes_file" : "../mimic/doc2vec_raw_training/training_sizes_{}.pkl",
    "training_events_sizes_labels_file" : "../mimic/doc2vec_raw_training/training_sizes_labels_{}.pkl",
    "testing_events_sizes_file" : "../mimic/doc2vec_raw_training/testing_sizes_{}.pkl",
    "testing_events_sizes_labels_file" : "../mimic/doc2vec_raw_training/testing_sizes_labels_{}.pkl",
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
    "gru": True,
    "useDropout": True,
    "dropout": 0.2,
    "trainingEpochs": 50,
    "batchSize": 16,
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
