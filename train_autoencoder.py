'''Example of VAE on MNIST dataset using MLP

The VAE has a modular design. The encoder, decoder and VAE
are 3 models that share weights. After training the VAE model,
the encoder can be used to generate latent vectors.
The decoder can be used to generate MNIST digits by sampling the
latent vector from a Gaussian distribution with mean = 0 and std = 1.

# Reference

[1] Kingma, Diederik P., and Max Welling.
"Auto-Encoding Variational Bayes."
https://arxiv.org/abs/1312.6114
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import numpy as np
import pandas as pd
import os

from sklearn.cross_validation import train_test_split

from model_creators import KerasVariationalAutoencoder
from normalization import NormalizationValues

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
inputShape = (None, len(aux.columns))
original_dim = len(aux.columns)
intermediate_dim = parameters['intermediate_dim']
latent_dim = parameters['latent_dim']

if __name__ == '__main__':
    autoencoder_creator = KerasVariationalAutoencoder((original_dim,),
                                                        intermediate_dim, latent_dim)
    autoencoder_adapter = autoencoder_creator.create()
    vae = autoencoder_adapter.vae
    encoder = autoencoder_adapter.encoder
    # train the autoencoder
    vae.fit(x_new,
            epochs=epochs,
            batch_size=batch_size)
    timeseries_vae, timeseries_encoder = autoencoder_creator.timedistribute_vae(input_shape, vae, encoder=encoder)
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