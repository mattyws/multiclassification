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

from adapter import KerasGeneratorAutoencoderAdapter
from data_generators import LongitudinalDataGenerator
from keras_callbacks import SaveModelEpoch
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
    # TODO: create generator
    print("===== Getting values for normalization =====")
    # normalization_values = Normalization.get_normalization_values(data[trainIndex])
    values = normalization_values.get_normalization_values(x_train,
                                                           saved_file_name="normalization_values_{}.pkl".format(i))
    normalizer = Normalization(values, temporary_path='data_tmp_autoencoder/')
    print("===== Normalizing fold data =====")
    normalizer.normalize_files(data)
    normalized_data = np.array(normalizer.get_new_paths(data))
    if not os.path.exists(parameters['trainingGeneratorPath']):
        dataTrainGenerator = LongitudinalDataGenerator(x_train,
                                                       y_train, parameters['batchSize'],
                                                       saved_batch_dir='training_batches_fold_{}'.format(i))
        dataTestGenerator = LongitudinalDataGenerator(x_test,
                                                      y_test, parameters['batchSize'],
                                                      saved_batch_dir='testing_batches_fold_{}'.format(i))
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

    autoencoder_creator = KerasVariationalAutoencoder((original_dim,),
                                                      intermediate_dim, latent_dim)
    if os.path.exists(parameters['modelCheckpointPath'] + 'autoencoder.model'):
        # TODO : load models
        autoencoder_adapter = KerasVariationalAutoencoder.create_from_path('autoencoder.model')
    else:
        # TODO: use adapter for training and prediction
        autoencoder_adapter = autoencoder_creator.create()

    modelCheckpoint = ModelCheckpoint(parameters['modelCheckpointPath'] + 'autoencoder.model')
    # train the autoencoder
    # def fit(self, dataGenerator, epochs=1, batch_size=10, workers=2, validationDataGenerator=None,
    #         validationSteps=None, callbacks=None):
    autoencoder_adapter.fit(x_train,
            epochs=epochs,
            batch_size=batch_size, callbacks=[modelCheckpoint])
    timeseries_vae, timeseries_encoder = autoencoder_creator.timedistribute_vae(input_shape, autoencoder_adapter.vae,
                                                                                encoder=autoencoder_adapter.encoder)
    results = timeseries_vae.predict(x_train)
    print("real", x_train[0])
    print("predicted", results[0])
    print("Encoder")
    results = timeseries_encoder.predict(x_train)
    print(results)

    # plot_results(models,
    #              data,
    #              batch_size=batch_size,
    #              model_name="vae_mlp")