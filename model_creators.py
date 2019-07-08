import abc

from keras.layers.wrappers import TimeDistributed
from keras.models import load_model
from keras.layers.core import Dense, Dropout, RepeatVector
from keras.layers.recurrent import LSTM, GRU
from keras.models import Sequential
from keras.layers import Lambda, Input, Dense
from keras.models import Model
from keras.datasets import mnist
from keras.losses import mse, binary_crossentropy
from keras.utils import plot_model
from keras import backend as K
from sklearn.neural_network.multilayer_perceptron import MLPClassifier

import adapter

class ModelCreator(object, metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def create(self):
        raise NotImplementedError('users must define \'create\' to use this base class')

class SimpleKerasRecurrentNNCreator(ModelCreator):

    def __init__(self, input_shape=None, numNeurouns=None, numOutputNeurons=None, activation='sigmoid', loss='categorical_crossentropy', optimizer='adam', use_dropout=False, dropout=0.5):
        self.input_shape = input_shape
        self.numNeurons = numNeurouns
        self.numOutputNeurons = numOutputNeurons
        self.activation = activation
        self.loss = loss
        self.optimizer = optimizer
        self.use_dropout = use_dropout
        self.dropout = dropout

    def __build_model(self):
        model = Sequential()
        model.add(LSTM(self.numNeurons, input_shape=self.input_shape))
        if self.use_dropout:
            model.add(Dropout(self.dropout))
        model.add(Dense(self.numOutputNeurons, activation='sigmoid'))
        model.compile(loss=self.loss, optimizer=self.optimizer, metrics=['accuracy'])
        return model

    def create(self):
        return adapter.KerasGeneratorAdapter(self.__build_model())

class MultilayerKerasRecurrentNNCreator(ModelCreator):
    def __init__(self, input_shape, outputUnits, numOutputNeurons,
                 layersActivations=None, networkActivation='sigmoid',
                 loss='categorical_crossentropy', optimizer='adam', gru=False, use_dropout=False, dropout=0.5,
                 metrics=['accuracy']):
        self.inputShape = input_shape
        self.outputUnits = outputUnits
        self.numOutputNeurons = numOutputNeurons
        self.networkActivation = networkActivation
        self.layersActivations = layersActivations
        self.loss = loss
        self.optimizer = optimizer
        self.gru = gru
        self.use_dropout = use_dropout
        self.dropout = dropout
        self.metrics = metrics

        self.__check_parameters()

    def __check_parameters(self):
        if self.layersActivations is not None and len(self.layersActivations) != len(self.outputUnits):
            raise ValueError("Output units must have the same size as activations!")

    def __build_model(self):
        model = Sequential()
        if len(self.outputUnits) > 1:
            for i in range(0, len(self.outputUnits)-1):
                if i == 0:
                    layer = self.__create_recurrent_layer(self.outputUnits[i], self.layersActivations[i], True,
                                                          inputShape=self.inputShape)
                else:
                    layer = self.__create_recurrent_layer(self.outputUnits[i], self.layersActivations[i], True)
                model.add(layer)
                if self.use_dropout:
                    model.add(Dropout(self.dropout))
        model.add(self.__create_recurrent_layer(self.outputUnits[-1], self.layersActivations[-1], False))
        if self.use_dropout:
            model.add(Dropout(self.dropout))
        model.add(Dense(self.numOutputNeurons, activation=self.networkActivation))
        model.compile(loss=self.loss, optimizer=self.optimizer, metrics=self.metrics)
        return model

    def __create_recurrent_layer(self, outputUnit, activation, returnSequences, inputShape=None):
        if self.gru:
            if inputShape:
                return GRU(outputUnit, activation=activation, input_shape=inputShape, return_sequences=returnSequences)
            else:
                return GRU(outputUnit, activation=activation, return_sequences=returnSequences)
        else:
            if inputShape:
                return LSTM(outputUnit, activation=activation, input_shape=inputShape, return_sequences=returnSequences)
            else:
                return LSTM(outputUnit, activation=activation, return_sequences=returnSequences)

    def create(self):
        return adapter.KerasGeneratorAdapter(self.__build_model())

    @staticmethod
    def create_from_path(filepath, custom_objects=None):
        model = load_model(filepath, custom_objects=custom_objects)
        return adapter.KerasGeneratorAdapter(model)

def sampling(args):
    """Reparameterization trick by sampling from an isotropic unit Gaussian.

    # Arguments
        args (tensor): mean and log of variance of Q(z|X)

    # Returns
        z (tensor): sampled latent vector
    """

    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean = 0 and std = 1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon

class KerasVariationalAutoencoder(ModelCreator):
    """
    A class that create a Variational Autoenconder.
    This script was writen based on https://github.com/keras-team/keras/blob/master/examples/variational_autoencoder.py
    See the linked script for more details on a tutorial of how to build it.
    """
    def __init__(self, input_shape, intermediate_dim, latent_dim, optmizer='adam', loss='mse', recurrent_autoencoder=False):
        self.input_shape = input_shape
        self.intermediate_dim = intermediate_dim
        self.latent_dim = latent_dim
        self.sampling = sampling
        self.loss = loss
        self.optmizer = optmizer
        self.recurrent_autoencoder = recurrent_autoencoder

    def create(self):
        if self.recurrent_autoencoder:
            encoder, decoder, vae = self.__build_recurrent_model()
        else:
            encoder, decoder, vae = self.__build_model()
        return adapter.KerasGeneratorAutoencoderAdapter(encoder, decoder, vae)

    def timedistribute_vae(self, input_shape, vae, encoder=None):
        timeseries_input = Input(shape=input_shape)
        vae = TimeDistributed(vae)(timeseries_input)
        vae = Model(timeseries_input, vae)
        if encoder is not None:
            encoder = TimeDistributed(encoder)(timeseries_input)
            encoder = Model(timeseries_input, encoder)
            return vae, encoder
        return vae

    def __build_model(self):
        # Encoder model
        inputs = Input(shape=self.input_shape, name='encoder_input')
        x = Dense(self.intermediate_dim, activation='relu')(inputs)
        z_mean = Dense(self.latent_dim, name='z_mean')(x)
        z_log_var = Dense(self.latent_dim, name='z_log_var')(x)
        z = Lambda(self.sampling, output_shape=(self.latent_dim,), name='z')([z_mean, z_log_var])
        encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
        # Decoder model
        latent_inputs = Input(shape=(self.latent_dim,), name='z_sampling')
        x = Dense(self.intermediate_dim, activation='relu')(latent_inputs)
        outputs = Dense(self.input_shape[0], activation='sigmoid')(x)
        decoder = Model(latent_inputs, outputs, name='decoder')
        # VAE
        outputs = decoder(encoder(inputs)[2])
        vae = Model(inputs, outputs, name='vae_mlp')
        vae.add_loss(self.__get_loss(inputs, outputs, z_mean, z_log_var))
        vae.compile(optimizer=self.optmizer)
        return encoder, decoder, vae

    def __build_recurrent_model(self):
        # Encoder
        inputs = Input(shape=self.input_shape, name='encoder_input')
        x = LSTM(self.intermediate_dim)(inputs)
        z_mean = Dense(self.latent_dim)(x)
        z_log_var = Dense(self.latent_dim)(x)
        # Z layer
        z = Lambda(self.sampling, name='z')([z_mean, z_log_var])
        # Decoder
        latent_inputs = RepeatVector(self.input_shape[0])(z)
        decoder_x = LSTM(self.intermediate_dim, return_sequences=True)(latent_inputs)
        outputs = LSTM(self.input_shape[1], return_sequences=True)(decoder_x)
        encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
        decoder = Model(latent_inputs, outputs, name='decoder')
        # VAE
        vae = Model(inputs, outputs, name='var')
        vae.add_loss(self.__get_loss(inputs, outputs, z_mean, z_log_var))
        vae.compile(optimizer=self.optmizer)
        return encoder, decoder, vae

    def __get_loss(self, inputs, outputs, z_mean, z_log):
        if self.loss == 'mse':
            reconstruction_loss = mse(inputs, outputs)
        else:
            reconstruction_loss = binary_crossentropy(inputs, outputs)
        reconstruction_loss *= self.input_shape[0]
        kl_loss = 1 + z_log - K.square(z_mean) - K.exp(z_log)
        kl_loss = K.sum(kl_loss, axis=-1)
        kl_loss *= -0.5
        vae_loss = K.mean(reconstruction_loss + kl_loss)
        return vae_loss


class SklearnNeuralNetwork(ModelCreator):

    def __init__(self, solver="lbfgs", alpha=1e-5, hidden_layer_sizes=10, random_state=1):
        self.solver = solver
        self.alpha = alpha
        self.hidden_layer_sizes = hidden_layer_sizes
        self.random_state = random_state

    def create(self):
        model = MLPClassifier(solver=self.solver, alpha=self.alpha, hidden_layer_sizes=self.hidden_layer_sizes, random_state=self.random_state)
        return adapter.SklearnAdapter(model)
