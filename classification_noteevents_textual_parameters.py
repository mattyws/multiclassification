from keras.layers import LeakyReLU

parameters = {
    "datasetCsvFilePath": "../mimic/new_dataset_patients.csv",
    "modelCheckpointPath": "../mimic/albert_raw_training/checkpoint/",
    "word2vecModelFileName": "../mimic/trained_doc2vec/50/doc2vec.model",
    "word2vec_representation_files_path": "../mimic/trained_doc2vec/50/transformed_representation/",
    "word2vec_padded_representation_files_path": "../mimic/albert_raw_training/padded_representation/",
    "embedding_size" : 50,
    "min_count" : 1,
    "workers" : 4,
    "window" : 4,
    "iterations" : 90,

    "modelConfigPath": "../mimic/albert_raw_training/checkpoint/config.json",
    "dataPath" : "../mimic/sepsis_noteevents_processed_ctakes/",
    "notes_word2vec_path" : "../mimic/sepsis_noteevents_preprocessed/",
    "trainingDataPath" :  "../mimic/albert_raw_training/dataTraining/",
    "testingDataPath" : "../mimic/albert_raw_training/dataTest/",
    "datasetFilesFileName": "../mimic/albert_raw_training/datasetFiles.pkl",
    "datasetLabelsFileName": "../mimic/albert_raw_training/datasetLabels.pkl",
    "trainingGeneratorPath": "../mimic/albert_raw_training/checkpoint/dataTrainGenerator.pkl",
    "testingGeneratorPath": "../mimic/albert_raw_training/checkpoint/dataTestGenerator.pkl",
    "resultFilePath": "../mimic/albert_raw_training/checkpoint/result.csv",
    "temporary_data_path" : "../mimic/albert_raw_training/data_tmp_{}/",
    "normalization_data_path": "../mimic/albert_raw_training/normalization_values_{}.pkl",
    "normalization_value_counts_path" : "../mimic/albert_raw_training/value_counts/",
    "training_events_sizes_file" : "../mimic/albert_raw_training/training_sizes_{}.pkl",
    "training_events_sizes_labels_file" : "../mimic/albert_raw_training/training_sizes_labels_{}.pkl",
    "testing_events_sizes_file" : "../mimic/albert_raw_training/testing_sizes_{}.pkl",
    "testing_events_sizes_labels_file" : "../mimic/albert_raw_training/testing_sizes_labels_{}.pkl",
    "dataLength" : 12,
    "outputUnits": [
        16
    ],
    "numOutputNeurons": 1,
    "loss": "binary_crossentropy",
    "optimizer":"adam",
    "layersActivations": [
        LeakyReLU()
    ],
    "networkActivation" : "sigmoid",
    "gru": True,
    "useDropout": True,
    "dropout": 0.5,
    "trainingEpochs": 40,
    "batchSize": 50,
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
