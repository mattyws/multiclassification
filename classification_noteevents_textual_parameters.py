from keras.layers import LeakyReLU

parameters = {
    "datasetCsvFilePath": "../mimic/new_dataset_patients.csv",
    "modelCheckpointPath": "../mimic/doc2vec_raw_training/checkpoint/",
    "word2vecModelFileName": "../mimic/doc2vec_raw_training/doc2vec.model",
    "word2vec_representation_files_path": "../mimic/doc2vec_raw_training/transformed_representation/",
    "word2vec_padded_representation_files_path": "../mimic/doc2vec_raw_training/padded_representation/",
    "embedding_size" : 200,
    "min_count" : 1,
    "workers" : 4,
    "window" : 4,
    "iterations" : 90,

    "modelConfigPath": "../mimic/doc2vec_raw_training/checkpoint/config.json",
    "dataPath" : "../mimic/textual_anonymized_data/",
    "notes_word2vec_path" : "../mimic/sepsis_noteevents_preprocessed/",
    "trainingDataPath" :  "../mimic/doc2vec_raw_training/dataTraining/",
    "testingDataPath" : "../mimic/doc2vec_raw_training/dataTest/",
    "datasetFilesFileName": "../mimic/doc2vec_raw_training/datasetFiles.pkl",
    "datasetLabelsFileName": "../mimic/doc2vec_raw_training/datasetLabels.pkl",
    "trainingGeneratorPath": "../mimic/doc2vec_raw_training/checkpoint/dataTrainGenerator.pkl",
    "testingGeneratorPath": "../mimic/doc2vec_raw_training/checkpoint/dataTestGenerator.pkl",
    "resultFilePath": "../mimic/doc2vec_raw_training/checkpoint/result.csv",
    "temporary_data_path" : "../mimic/doc2vec_raw_training/data_tmp_{}/",
    "normalization_data_path": "../mimic/doc2vec_raw_training/normalization_values_{}.pkl",
    "normalization_value_counts_path" : "../mimic/doc2vec_raw_training/value_counts/",
    "training_events_sizes_file" : "../mimic/doc2vec_raw_training/training_sizes_{}.pkl",
    "training_events_sizes_labels_file" : "../mimic/doc2vec_raw_training/training_sizes_labels_{}.pkl",
    "testing_events_sizes_file" : "../mimic/doc2vec_raw_training/testing_sizes_{}.pkl",
    "testing_events_sizes_labels_file" : "../mimic/doc2vec_raw_training/testing_sizes_labels_{}.pkl",
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
    "networkActivation" : "relu",
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
