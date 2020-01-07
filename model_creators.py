import abc

import keras
from keras.layers.convolutional import Conv2D, Conv3D
from keras.layers.wrappers import TimeDistributed
from keras.models import load_model
from keras.layers.core import Dense, Dropout, RepeatVector, Reshape
from keras.layers.recurrent import LSTM, GRU
from keras.models import Sequential
from keras.layers import Lambda, Input, Dense, Flatten, Conv1D, AveragePooling1D, GlobalAveragePooling1D, Concatenate, \
    GlobalAveragePooling2D, Masking, LeakyReLU
from keras.models import Model
from keras.datasets import mnist
from keras.losses import mse, binary_crossentropy
from keras.utils import plot_model
from keras import backend as K

import adapter
from adapter import KerasAutoencoderAdapter


def create_recurrent_layer(self, outputUnit, activation, returnSequences, inputShape=None):
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

class ModelCreator(object, metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def create(self):
        raise NotImplementedError('users must define \'create\' to use this base class')


class NoteeventsClassificationModelCreator(ModelCreator):

    def __init__(self, input_shape, outputUnits, numOutputNeurons, embedding_size = None,
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
        self.embedding_size = embedding_size
        self.__check_parameters()

    def __check_parameters(self):
        if self.layersActivations is not None and len(self.layersActivations) != len(self.outputUnits):
            raise ValueError("Output units must have the same size as activations!")

    def create(self, model_summary_filename=None):
        input, output = self.build_network()
        model = Model(inputs=input, outputs=output)
        model.compile(loss=self.loss, optimizer=self.optimizer, metrics=self.metrics)
        print(model.summary())
        if model_summary_filename is not None:
            with open(model_summary_filename, 'w') as summary_file:
                model.summary(print_fn=lambda x: summary_file.write(x + '\n'))
        return adapter.KerasGeneratorAdapter(model)

    def build_network(self):
        representation_model = Sequential()
        representation_model.add(Masking(mask_value=0., name="representation_masking"))
        representation_model.add(LSTM(64, dropout=.3, name="representation_lstm"))
        representation_model.add(LeakyReLU(alpha=.3, name="representation_leakyrelu"))
        representation_model.add(Dense(32, name="representation_dense"))
        input = Input(self.inputShape)
        layer = TimeDistributed(representation_model, name="representation_model")(input)
        if len(self.outputUnits) == 1:
            layer = self.__create_recurrent_layer(self.outputUnits[0], self.layersActivations[0], False)(layer)
        else:
            layer = self.__create_recurrent_layer(self.outputUnits[0], self.layersActivations[0], True)(layer)
        if len(self.outputUnits) > 1:
            for i in range(1, len(self.outputUnits)):
                if self.use_dropout:
                    dropout = Dropout(self.dropout)(layer)
                    layer = dropout
                layer = self.__create_recurrent_layer(self.outputUnits[i], self.layersActivations[i], True)(layer)
        if self.use_dropout:
            dropout = Dropout(self.dropout)(layer)
            layer = dropout
        output = Dense(self.numOutputNeurons, activation=self.networkActivation)(layer)
        return input, output

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


class EnsembleModelCreator(ModelCreator):

    def __init__(self, level_zero_pmodels, pinput_shape, level_zero_smodels=None, sinput_shape=None,
                 level_one_model_output_units=None, level_one_use_droupout=False, level_one_dropout=.5,
                 level_one_model_layers_activation=None):
        if (level_zero_smodels is None and sinput_shape is not None) \
                or (level_zero_smodels is not None and sinput_shape is None):
            raise ValueError("") #todo error message
        self.level_zero_pmodels = level_zero_pmodels
        self.level_zero_smodels = level_zero_smodels
        self.pinput_shape = pinput_shape
        self.sinput_shape = sinput_shape
        self.level_one_model_output_units = level_one_model_output_units
        self.level_one_use_droupout = level_one_use_droupout
        self.level_one_dropout = level_one_dropout
        self.level_one_model_layers_activation = level_one_model_layers_activation

    def create(self):
        return self.build_model()

    def build_model(self):
        # Inputs and outputs for the type of data
        inputs = []
        outputs = []
        pinput, poutputs = self.__create_models_layer(self.level_zero_pmodels, self.pinput_shape)
        inputs.append(pinput)
        outputs.extend(poutputs)
        if self.level_zero_smodels is not None and self.sinput_shape is not None:
            sinput, soutputs = self.__create_models_layer(self.level_zero_smodels, self.sinput_shape)
            inputs.append(sinput)
            outputs.extend(soutputs)
        concat = Concatenate(outputs)
        # TODO: construir o resto do modelo
        if len(self.level_one_model_output_units) == 1:
            layer = Dense(self.level_one_model_output_units[0], self.level_one_model_layers_activation[0], False)(input)
        else:
            layer = self.__create_recurrent_layer(self.outputUnits[0], self.layersActivations[0], True)(input)
        if len(self.outputUnits) > 1:
            for i in range(1, len(self.outputUnits)):
                if self.use_dropout:
                    dropout = Dropout(self.dropout)(layer)
                    layer = dropout
                if i == len(self.outputUnits) - 1:
                    layer = self.__create_recurrent_layer(self.outputUnits[i], self.layersActivations[i], False)(layer)
                else:
                    layer = self.__create_recurrent_layer(self.outputUnits[i], self.layersActivations[i], True)(layer)
        if self.use_dropout:
            dropout = Dropout(self.dropout)(layer)
            layer = dropout

        output = Dense(self.numOutputNeurons, activation=self.networkActivation,
                       kernel_regularizer=self.kernel_regularizer, bias_regularizer=self.bias_regularizer,
                       activity_regularizer=self.activity_regularizer)(layer)
        return input, output

    def __create_models_layer(self, models, input_shape):
        input = Input(input_shape)
        outputs = []
        for model in models:
            out = model(input)
            new_model = Model(inputs=input, outputs=model.layers[-2].get_output_at(0), trainable=False)
            outputs.append(new_model.output)
        return input, outputs


class MultilayerKerasRecurrentNNCreator(ModelCreator):
    def __init__(self, input_shape, outputUnits, numOutputNeurons,
                 layersActivations=None, networkActivation='sigmoid',
                 loss='categorical_crossentropy', optimizer='adam', gru=False, use_dropout=False, dropout=0.5,
                 metrics=['accuracy'], kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None):
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
        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer
        self.activity_regularizer = activity_regularizer

        self.__check_parameters()

    def __check_parameters(self):
        if self.layersActivations is not None and len(self.layersActivations) != len(self.outputUnits):
            raise ValueError("Output units must have the same size as activations!")

    def build_network(self):
        input = Input(self.inputShape)
        if len(self.outputUnits) == 1:
            layer = self.__create_recurrent_layer(self.outputUnits[0], self.layersActivations[0], False)(input)
        else:
            layer = self.__create_recurrent_layer(self.outputUnits[0], self.layersActivations[0], True)(input)
        if len(self.outputUnits) > 1:
            for i in range(1, len(self.outputUnits)):
                if self.use_dropout:
                    dropout = Dropout(self.dropout)(layer)
                    layer = dropout
                if i == len(self.outputUnits) - 1:
                    layer = self.__create_recurrent_layer(self.outputUnits[i], self.layersActivations[i], False)(layer)
                else:
                    layer = self.__create_recurrent_layer(self.outputUnits[i], self.layersActivations[i], True)(layer)
        if self.use_dropout:
            dropout = Dropout(self.dropout)(layer)
            layer = dropout
        output = Dense(self.numOutputNeurons, activation=self.networkActivation,
                       kernel_regularizer=self.kernel_regularizer, bias_regularizer=self.bias_regularizer,
                       activity_regularizer=self.activity_regularizer)(layer)
        return input, output

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

    def create(self, model_summary_filename=None):
        input, output = self.build_network()
        model = Model(inputs=input, outputs=output)
        model.compile(loss=self.loss, optimizer=self.optimizer, metrics=self.metrics)
        if model_summary_filename is not None:
            with open(model_summary_filename, 'w') as summary_file:
                model.summary(print_fn=lambda x: summary_file.write(x + '\n'))
        return adapter.KerasGeneratorAdapter(model)

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
        if len(self.outputUnits) == 1:
            model.add(self.__create_recurrent_layer(self.outputUnits[-1], self.layersActivations[-1], False,
                                                    inputShape=self.inputShape))
        else:
            model.add(self.__create_recurrent_layer(self.outputUnits[-1], self.layersActivations[-1], False))
        if self.use_dropout:
            model.add(Dropout(self.dropout))
        model.add(Dense(self.numOutputNeurons, activation=self.networkActivation))
        model.compile(loss=self.loss, optimizer=self.optimizer, metrics=self.metrics)
        return model

    def create_sequential(self, model_summary_filename=None):
        model = self.__build_model()
        if model_summary_filename is not None:
            with open(model_summary_filename, 'w') as summary_file:
                model.summary(print_fn=lambda x: summary_file.write(x + '\n'))
        return adapter.KerasGeneratorAdapter(model)

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
        return adapter.KerasAutoencoderAdapter(encoder, decoder, vae)

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
        # TODO: não aceita retorno de um vetor [zmeean, ...], ver qual saída uso para o encoder
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
        kl_loss *=.8
        vae_loss = K.mean(reconstruction_loss + kl_loss)
        return vae_loss

    @staticmethod
    def create_from_path(filename):
        encoder = load_model('encoder_' + filename)
        decoder = load_model('decoder_' + filename)
        vae = load_model(filename)
        return KerasAutoencoderAdapter(encoder, decoder, vae)
