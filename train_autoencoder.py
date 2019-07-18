from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import pickle

import numpy as np
import pandas as pd
import os

from keras.callbacks import ModelCheckpoint
from sklearn.cross_validation import train_test_split

from adapter import KerasAutoencoderAdapter
from data_generators import LongitudinalDataGenerator, AutoencoderDataGenerator
from keras_callbacks import ResumeTrainingCallback
from model_creators import KerasVariationalAutoencoder
from normalization import NormalizationValues, Normalization

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
DATETIME_PATTERN = "%Y-%m-%d %H:%M:%S"

parametersFilePath = "./autoencoder_parameters.json"

#Loading parameters file
print("========= Loading Parameters")
parameters = None
with open(parametersFilePath, 'r') as parametersFileHandler:
    parameters = json.load(parametersFileHandler)
if parameters is None:
    exit(1)

if not os.path.exists(parameters['modelCheckpointPath']):
    os.mkdir(parameters['modelCheckpointPath'])

# Loading csv
print("========= Loading data")
data_csv = pd.read_csv(parameters['datasetCsvFilePath'])
data_csv = data_csv.sort_values(['icustay_id'])
# Get the values in data_csv that have events saved
data = np.array([itemid for itemid in list(data_csv['icustay_id'])
                 if os.path.exists(parameters['dataPath'] + '{}.csv'.format(itemid))])
data_csv = data_csv[data_csv['icustay_id'].isin(data)]
data = np.array([parameters['dataPath'] + '{}.csv'.format(itemid) for itemid in data])
print("========= Transforming classes")
classes = np.array([1 if c == 'sepsis' else 1 for c in list(data_csv['class'])])


x_train, x_test, y_train, y_test = train_test_split(data, classes,
                                                    stratify=classes,
                                                    test_size=0.20)

print("========= Preparing normalization values")
normalization_values = NormalizationValues(data)
normalization_values.prepare()
# Get input shape
aux = pd.read_csv(data[0])
if 'Unnamed: 0' in aux.columns:
    aux = aux.drop(columns=['Unnamed: 0'])
if 'chartevents_Unnamed: 0' in aux.columns:
    aux = aux.drop(columns=['chartevents_Unnamed: 0'])
if 'labevents_Unnamed: 0' in aux.columns:
    aux = aux.drop(columns=['labevents_Unnamed: 0'])
input_shape = (None, len(aux.columns))
original_dim = len(aux.columns)
intermediate_dim = parameters['intermediate_dim']
latent_dim = parameters['latent_dim']
epochs = parameters['epochs']
batch_size = parameters['batch_size']

config = None
if os.path.exists(parameters['modelConfigPath']):
    with open(parameters['modelConfigPath'], 'r') as configHandler:
        config = json.load(configHandler)

if __name__ == '__main__':
    print("===== Getting values for normalization =====")
    # normalization_values = Normalization.get_normalization_values(data[trainIndex])
    values = normalization_values.get_normalization_values(x_train,
                                                           saved_file_name="normalization_values_autoencoder.pkl")
    normalizer = Normalization(values, temporary_path='data_tmp_autoencoder/')
    print("===== Normalizing fold data =====")
    normalizer.normalize_files(data)
    normalized_data = np.array(normalizer.get_new_paths(data))
    if not os.path.exists(parameters['trainingGeneratorPath']):
        dataTrainGenerator = AutoencoderDataGenerator(x_train)
        dataTestGenerator = AutoencoderDataGenerator(x_test)
        print("========= Saving generators")
        with open(parameters['trainingGeneratorPath'], 'wb') as trainingGeneratorHandler:
            pickle.dump(dataTrainGenerator, trainingGeneratorHandler, pickle.HIGHEST_PROTOCOL)

        with open(parameters['testingGeneratorPath'], 'wb') as testingGeneratorHandler:
            pickle.dump(dataTestGenerator, testingGeneratorHandler, pickle.HIGHEST_PROTOCOL)
    else:
        print("========= Loading generators")
        with open(parameters['trainingGeneratorPath'], 'rb') as trainingGeneratorHandler:
            dataTrainGenerator = pickle.load(trainingGeneratorHandler)

        with open(parameters['testingGeneratorPath'], 'rb') as testingGeneratorHandler:
            dataTestGenerator = pickle.load(testingGeneratorHandler)

    if config is not None:
        configSaver = ResumeTrainingCallback(parameters['modelConfigPath'],
                                             parameters['modelCheckpointPath'] + 'autoencoder.model', 0,
                                             alreadyTrainedEpochs=config['epoch'])
    else:
        configSaver = ResumeTrainingCallback(parameters['modelConfigPath'],
                                             parameters['modelCheckpointPath'] + 'autoencoder.model', 0)
    autoencoder_creator = KerasVariationalAutoencoder((original_dim,),
                                                      intermediate_dim, latent_dim)
    if os.path.exists(parameters['modelCheckpointPath'] + 'autoencoder.model'):
        autoencoder_adapter = KerasVariationalAutoencoder.create_from_path('autoencoder.model')
    else:
        autoencoder_adapter = autoencoder_creator.create()

    modelCheckpoint = ModelCheckpoint(parameters['modelCheckpointPath'] + 'autoencoder.model')
    autoencoder_adapter.fit(x_train,
            epochs=epochs, validationDataGenerator=dataTestGenerator,
            steps_per_epoch=len(data), callbacks=[modelCheckpoint, configSaver])
    autoencoder_adapter.save(parameters['modelCheckpointPath'] + 'autoencoder_final.model')
    print("{} events in database".format(dataTrainGenerator.total_events + dataTestGenerator.total_events))