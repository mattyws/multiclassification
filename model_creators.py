import abc
import copy

import bert
import tensorflow
from kerastuner import HyperModel
from kerastuner.engine.tuner import Tuner
from tensorflow.keras.regularizers import l1_l2
from tensorflow import keras
from tensorflow.keras.layers import TimeDistributed
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense, Dropout, RepeatVector, Reshape
from tensorflow.keras.layers import LSTM, GRU
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Lambda, Input, Dense, Flatten, Conv1D, AveragePooling1D, GlobalAveragePooling1D, Concatenate, \
    GlobalAveragePooling2D, Masking, LeakyReLU, MaxPool1D
from tensorflow.keras.models import Model
from tensorflow.keras.losses import mse, binary_crossentropy
from tensorflow.keras import backend as K
from tcn import TCN
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.backend import concatenate
from tensorflow.python.keras.layers import LeakyReLU
from tensorflow.python.keras.optimizer_v2.adam import Adam
from tensorflow.python.keras.utils.vis_utils import plot_model

import adapter
from adapter import KerasAutoencoderAdapter
import os


def create_recurrent_layer(outputUnit, activation='tanh', returnSequences=None, gru=False):
    if gru:
        return GRU(outputUnit, activation=activation, return_sequences=returnSequences)
    else:
        return LSTM(outputUnit, activation=activation, return_sequences=returnSequences)

class ModelCreator(object, metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def create(self):
        raise NotImplementedError('users must define \'create\' to use this base class')


class KerasTunerModelCreator(ModelCreator):

    def __init__(self, tuner : Tuner, hyper_model:HyperModel, model_name : str, remove_last_layer: bool = False):
        self.tuner = tuner
        self.hyper_model = hyper_model
        self.best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]
        self.name = model_name
        self.remove_last_layer = remove_last_layer

    def create(self, model_summary_filename=None):
        # model = self.tuner.hypermodel.build(self.best_hp)
        model = self.hyper_model.build(self.best_hp)
        print(model.summary())
        if model_summary_filename is not None:
            with open(model_summary_filename, 'w') as summary_file:
                model.summary(print_fn=lambda x: summary_file.write(x + '\n'))
        if self.remove_last_layer:
            model = self.remove_last_model_layer(model)
        return adapter.KerasAdapter(model)

    def remove_last_model_layer(self, model):
        print(model.layers)
        # exit()
        new_model = Model(inputs=model.input, outputs=model.layers[-2].output)
        new_model.compile(loss=model.loss, optimizer=model.optimizer)
        print(new_model.summary())
        return new_model

class BertModelCreator(object):

    def __init__(self, input_shape, albert_model_dir=None):
        self.albert_model_dir = albert_model_dir
        self.input_shape = input_shape

    def create_from_model_dir(self, model_dir, cptk_name):
        bert_params = bert.params_from_pretrained_ckpt(model_dir)

        model, l_bert = self.build_model(bert_params)

        bert_ckpt_file = os.path.join(model_dir, cptk_name)
        bert.load_stock_weights(l_bert, bert_ckpt_file)

        bert.load_albert_weights(l_bert, bert_ckpt_file)
        return model, l_bert

    def create_representation_model(self, albert_model_name = "albert_base_v2"):
        albert_dir = bert.fetch_google_albert_model(albert_model_name, ".models")
        model_ckpt = os.path.join(albert_dir, "model.ckpt-best")

        albert_params = bert.albert_params(albert_dir)
        model, l_bert = self.build_model(albert_params)

        bert.load_albert_weights(l_bert, model_ckpt)



        # albert_model_name = self.albert_model_dir #os.path.join(self.albert_model_dir, albert_model_name)
        # albert_dir = bert.fetch_tfhub_albert_model(albert_model_name, ".models")
        #
        # albert_params = bert.albert_params(albert_model_name)
        # model, l_bert = self.build_model(albert_params)
        #
        # bert.load_albert_weights(l_bert, albert_dir)
        return model, l_bert

    def build_model(self, bert_params):
        l_bert = bert.BertModelLayer.from_params(bert_params, name="bert")
        # model = keras.Sequential([
        #     l_bert,
        #     Lambda(lambda x: x[:, -0, ...]),  # [B, 2]
        #     Dense(units=1, activation="softmax"),  # [B, 10, 2]
        # ])

        l_input_ids = keras.layers.Input(shape=self.input_shape[-1])
        output = l_bert(l_input_ids)
        output = keras.layers.Lambda(lambda x: x[:, 0, :])(output)
        output = Dropout(0.3)(output)
        output = keras.layers.Dense(1, activation="sigmoid", kernel_regularizer=l1_l2())(output)
        model = keras.Model(inputs=l_input_ids, outputs=output)

        model.build(input_shape=self.input_shape)
        model.compile(optimizer=Adam(),
                      loss="binary_crossentropy",
                      metrics=[keras.metrics.SparseCategoricalAccuracy(name="acc")])

        for weight in l_bert.weights:
            print(weight.name)

        return model, l_bert

class NoteeventsClassificationModelCreator(ModelCreator):

    def __init__(self, input_shape, outputUnits, numOutputNeurons, embedding_size = None,
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
        self.embedding_size = embedding_size
        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer
        self.activity_regularizer = activity_regularizer
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
        return adapter.KerasAdapter(model)

    def build_network(self):
        representation_model = Sequential()
        representation_model.add(Masking(mask_value=0., name="representation_masking"))
        representation_model.add(GRU(64, dropout=.3, name="representation_gru"))
        representation_model.add(LeakyReLU(alpha=.3, name="representation_leakyrelu"))
        representation_model.add(Dense(32, name="representation_dense"))
        input = Input(self.inputShape)
        layer = TimeDistributed(representation_model, name="representation_model")(input)
        if len(self.outputUnits) == 1:
            layer = create_recurrent_layer(self.outputUnits[0], returnSequences=False
                                           , gru=self.gru)(layer)
        else:
            layer = create_recurrent_layer(self.outputUnits[0], returnSequences=True
                                           , gru=self.gru)(layer)
        activation = copy.deepcopy(self.layersActivations[0])
        layer = activation(layer)
        if len(self.outputUnits) > 1:
            for i in range(1, len(self.outputUnits)):
                if self.use_dropout:
                    dropout = Dropout(self.dropout)(layer)
                    layer = dropout
                if i == len(self.outputUnits) - 1:
                    layer = create_recurrent_layer(self.outputUnits[i], returnSequences=False
                                                   , gru=self.gru)(layer)
                else:
                    layer = create_recurrent_layer(self.outputUnits[i], returnSequences=True
                                                   , gru=self.gru)(layer)
                activation = copy.deepcopy(self.layersActivations[i])
                layer = activation(layer)
        if self.use_dropout:
            dropout = Dropout(self.dropout)(layer)
            layer = dropout
        output = Dense(self.numOutputNeurons, activation=self.networkActivation,
                       kernel_regularizer=self.kernel_regularizer, bias_regularizer=self.bias_regularizer,
                       activity_regularizer=self.activity_regularizer)(layer)
        return input, output


class NoteeventsClassificationTCNModelCreator(ModelCreator):

    def __init__(self, input_shape, outputUnits, numOutputNeurons, embedding_size = None,
                 layersActivations=None, networkActivation='sigmoid', pooling=None, kernel_sizes=None,
                 loss='categorical_crossentropy', optimizer='adam', gru=False, use_dropout=False, dropout=0.5,
                 dilations=[[1, 2, 4]], nb_stacks=[1],
                 metrics=['accuracy'], kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None):
        self.inputShape = input_shape
        self.outputUnits = outputUnits
        self.numOutputNeurons = numOutputNeurons
        self.networkActivation = networkActivation
        self.layersActivations = layersActivations
        self.loss = loss
        self.optimizer = optimizer
        self.gru = gru
        self.kernel_sizes = kernel_sizes
        self.dilatations = dilations
        self.nb_stacks = nb_stacks
        self.pooling = pooling
        self.use_dropout = use_dropout
        self.dropout = dropout
        self.metrics = metrics
        self.embedding_size = embedding_size
        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer
        self.activity_regularizer = activity_regularizer
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
        return adapter.KerasAdapter(model)

    def build_network(self):
        representation_model = Sequential()
        representation_model.add(Masking(mask_value=0., name="representation_masking"))
        representation_model.add(GRU(64, dropout=.3, name="representation_gru"))
        representation_model.add(LeakyReLU(alpha=.3, name="representation_leakyrelu"))
        representation_model.add(Dense(32, name="representation_dense"))
        input = Input(self.inputShape)
        layer = TimeDistributed(representation_model, name="representation_model")(input)

        if len(self.outputUnits) == 1:
            layer = TCN(self.outputUnits[0], kernel_size=self.kernel_sizes[0], dilations=self.dilatations[0]
                        , nb_stacks=self.nb_stacks[0], return_sequences=False)(layer)
        else:
            layer = TCN(self.outputUnits[0], kernel_size=self.kernel_sizes[0], dilations=self.dilatations[0],
                        nb_stacks=self.nb_stacks[0], return_sequences=True)(input)
        activation = copy.deepcopy(self.layersActivations[0])
        layer = activation(layer)
        if self.pooling[0]:
            layer = MaxPool1D()(layer)
        if len(self.outputUnits) > 1:
            for i in range(1, len(self.outputUnits)):
                if i == len(self.outputUnits) - 1:
                    layer = TCN(self.outputUnits[i], kernel_size=self.kernel_sizes[i], dilations=self.dilatations[i],
                                nb_stacks=self.nb_stacks[i], return_sequences=False)(input)
                else:
                    layer = TCN(self.outputUnits[i], kernel_size=self.kernel_sizes[i], dilations=self.dilatations[i],
                                nb_stacks=self.nb_stacks[i], return_sequences=True)(input)
                activation = copy.deepcopy(self.layersActivations[i])
                layer = activation(layer)
                if self.pooling[i]:
                    layer = MaxPool1D()(layer)
        if self.use_dropout:
            dropout = Dropout(self.dropout)(layer)
            layer = dropout
        output = Dense(self.numOutputNeurons, activation=self.networkActivation,
                       kernel_regularizer=self.kernel_regularizer, bias_regularizer=self.bias_regularizer,
                       activity_regularizer=self.activity_regularizer)(layer)
        return input, output


class EnsembleModelCreator(ModelCreator):

    def __init__(self, input_shape, num_output_neurons, output_units=None, use_dropout=False, dropout=.5,
                 layers_activation=None, network_activation='sigmoid', loss='categorical_crossentropy', optimizer='adam',
                 metrics=['accuracy'], kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None):
        self.input_shape = input_shape
        self.num_output_neurons = num_output_neurons
        self.output_units = output_units
        self.use_dropout = use_dropout
        self.dropout = dropout
        self.layers_activation = layers_activation
        self.network_activation = network_activation
        self.loss = loss
        self.optimizer = optimizer
        self.metrics = metrics
        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer
        self.activity_regularizer = activity_regularizer

    def create(self, model_summary_filename=None):
        input, output = self.build_network()
        model = Model(inputs=input, outputs=output)
        model.compile(loss=self.loss, optimizer=self.optimizer, metrics=self.metrics)
        if model_summary_filename is not None:
            with open(model_summary_filename, 'w') as summary_file:
                model.summary(print_fn=lambda x: summary_file.write(x + '\n'))
        return adapter.KerasAdapter(model)

    def build_network(self):
        input = Input(shape=self.input_shape)
        layer = Dense(self.output_units[0])(input)
        activation = copy.deepcopy(self.layers_activation[0])
        layer = activation(layer)
        if len(self.output_units) > 1:
            for i in range(1, len(self.output_units)):
                if self.use_dropout:
                    dropout = Dropout(self.dropout)(layer)
                    layer = dropout
                layer = Dense(self.output_units[i])(layer)
                activation = copy.deepcopy(self.layers_activation[i])
                layer = activation(layer)
        if self.use_dropout:
            dropout = Dropout(self.dropout)(layer)
            layer = dropout
        output = Dense(self.num_output_neurons, activation=self.network_activation,
                       kernel_regularizer=self.kernel_regularizer, bias_regularizer=self.bias_regularizer,
                       activity_regularizer=self.activity_regularizer)(layer)
        return input, output



class MultilayerKerasRecurrentNNCreator(ModelCreator):
    def __init__(self, input_shape, outputUnits, numOutputNeurons,
                 layersActivations=None, networkActivation='sigmoid',
                 loss='categorical_crossentropy', optimizer='adam',gru=False, use_dropout=False, dropout=0.5,
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
        if gru:
            self.name = "GRU_MODEL"
        else:
            self.name = "LSTM_MODEL"

    def __check_parameters(self):
        if self.layersActivations is not None and len(self.layersActivations) != len(self.outputUnits):
            raise ValueError("Output units must have the same size as activations!")

    def build_network(self):
        input = Input(self.inputShape)
        if len(self.outputUnits) == 1:
            layer = create_recurrent_layer(self.outputUnits[0], returnSequences=False
                                           , gru=self.gru)(input)
        else:
            layer = create_recurrent_layer(self.outputUnits[0], returnSequences=True
                                           , gru=self.gru)(input)
        activation = copy.deepcopy(self.layersActivations[0])
        layer = activation(layer)
        if len(self.outputUnits) > 1:
            for i in range(1, len(self.outputUnits)):
                if self.use_dropout:
                    dropout = Dropout(self.dropout)(layer)
                    layer = dropout
                if i == len(self.outputUnits) - 1:
                    layer = create_recurrent_layer(self.outputUnits[i], returnSequences=False
                                                   , gru=self.gru)(layer)
                else:
                    layer = create_recurrent_layer(self.outputUnits[i], returnSequences=True
                                                   , gru=self.gru)(layer)
                activation = copy.deepcopy(self.layersActivations[i])
                layer = activation(layer)
        if self.use_dropout:
            dropout = Dropout(self.dropout)(layer)
            layer = dropout
        if self.numOutputNeurons is not None:
            output = Dense(self.numOutputNeurons, activation=self.networkActivation,
                           kernel_regularizer=self.kernel_regularizer, bias_regularizer=self.bias_regularizer,
                           activity_regularizer=self.activity_regularizer)(layer)
        else:
            output = layer
        return input, output

    def create(self, model_summary_filename=None):
        input, output = self.build_network()
        model = Model(inputs=input, outputs=output)
        model.compile(loss=self.loss, optimizer=self.optimizer, metrics=self.metrics)
        if model_summary_filename is not None:
            with open(model_summary_filename, 'w') as summary_file:
                model.summary(print_fn=lambda x: summary_file.write(x + '\n'))
        return adapter.KerasAdapter(model)

    @staticmethod
    def create_from_path(filepath, custom_objects=None):
        model = load_model(filepath, custom_objects=custom_objects)
        return adapter.KerasAdapter(model)

class MultilayerConvolutionalNNCreator(ModelCreator):
    def __init__(self, input_shape, outputUnits, numOutputNeurons,
                 layersActivations=None, networkActivation='sigmoid', pooling=None, kernel_sizes=None,
                 loss='categorical_crossentropy', optimizer='adam', use_dropout=False, dropout=0.5,
                 metrics=['accuracy'], kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None):
        self.inputShape = input_shape
        self.outputUnits = outputUnits
        self.numOutputNeurons = numOutputNeurons
        self.networkActivation = networkActivation
        self.layersActivations = layersActivations
        self.kernel_sizes = kernel_sizes
        self.loss = loss
        self.pooling = pooling
        self.optimizer = optimizer
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
            layer = Conv1D(self.outputUnits[0], kernel_size=self.kernel_sizes[0])(input)
        else:
            layer = Conv1D(self.outputUnits[0], kernel_size=self.kernel_sizes[0])(input)
        activation = copy.deepcopy(self.layersActivations[0])
        layer = activation(layer)
        if self.pooling[0]:
            layer = GlobalAveragePooling1D()(layer)
        if len(self.outputUnits) > 1:
            for i in range(1, len(self.outputUnits)):
                if i == len(self.outputUnits) - 1:
                    layer = Conv1D(self.outputUnits[i], kernel_size=self.kernel_sizes[i], return_sequences=False)(input)
                else:
                    layer = Conv1D(self.outputUnits[i], kernel_size=self.kernel_sizes[i], return_sequences=True)(input)
                activation = copy.deepcopy(self.layersActivations[i])
                layer = activation(layer)
                if self.pooling[i]:
                    layer = MaxPool1D()(layer)
        if self.use_dropout:
            dropout = Dropout(self.dropout)(layer)
            layer = dropout
        output = Dense(self.numOutputNeurons, activation=self.networkActivation,
                       kernel_regularizer=self.kernel_regularizer, bias_regularizer=self.bias_regularizer,
                       activity_regularizer=self.activity_regularizer)(layer)
        return input, output

    def create(self, model_summary_filename=None):
        input, output = self.build_network()
        model = Model(inputs=input, outputs=output)
        model.compile(loss=self.loss, optimizer=self.optimizer, metrics=self.metrics)
        if model_summary_filename is not None:
            with open(model_summary_filename, 'w') as summary_file:
                model.summary(print_fn=lambda x: summary_file.write(x + '\n'))
        return adapter.KerasAdapter(model)

    @staticmethod
    def create_from_path(filepath, custom_objects=None):
        model = load_model(filepath, custom_objects=custom_objects)
        return adapter.KerasAdapter(model)


class MultilayerTemporalConvolutionalNNCreator(ModelCreator):
    def __init__(self, input_shape, outputUnits, numOutputNeurons,
                 layersActivations=None, networkActivation='sigmoid', pooling=None, kernel_sizes=None,
                 loss='categorical_crossentropy', optimizer='adam', use_dropout=False, dropout=0.5,
                 dilations=[[1, 2, 4]], nb_stacks=[1], model_input_name:str=None,
                 metrics=['accuracy'], kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None):
        self.inputShape = input_shape
        self.outputUnits = outputUnits
        self.numOutputNeurons = numOutputNeurons
        self.networkActivation = networkActivation
        self.layersActivations = layersActivations
        self.kernel_sizes = kernel_sizes
        self.loss = loss
        self.dilatations = dilations
        self.nb_stacks = nb_stacks
        self.pooling = pooling
        self.optimizer = optimizer
        self.use_dropout = use_dropout
        self.dropout = dropout
        self.metrics = metrics
        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer
        self.activity_regularizer = activity_regularizer
        self.__check_parameters()
        self.name = "TCN_MODEL"
        self.model_input_name = model_input_name

    def __check_parameters(self):
        if self.layersActivations is not None and len(self.layersActivations) != len(self.outputUnits):
            raise ValueError("Output units must have the same size as activations!")

    def build_network(self):
        if self.model_input_name is None:
            input = Input(self.inputShape)
        else:
            input = Input(self.inputShape, name=self.model_input_name)
        if len(self.outputUnits) == 1:
            layer = TCN(self.outputUnits[0], kernel_size=self.kernel_sizes[0], dilations=self.dilatations[0]
                        , nb_stacks=self.nb_stacks[0], return_sequences=False)(input)
        else:
            layer = TCN(self.outputUnits[0], kernel_size=self.kernel_sizes[0], dilations=self.dilatations[0],
                        nb_stacks=self.nb_stacks[0], return_sequences=True)(input)
        activation = copy.deepcopy(self.layersActivations[0])
        layer = activation(layer)
        if self.pooling[0]:
            layer = MaxPool1D()(layer)
        if len(self.outputUnits) > 1:
            for i in range(1, len(self.outputUnits)):
                if i == len(self.outputUnits) - 1:
                    layer = TCN(self.outputUnits[i], kernel_size=self.kernel_sizes[i], dilations=self.dilatations[i],
                                nb_stacks=self.nb_stacks[i], return_sequences=False)(input)
                else:
                    layer = TCN(self.outputUnits[i], kernel_size=self.kernel_sizes[i], dilations=self.dilatations[i],
                                nb_stacks=self.nb_stacks[i], return_sequences=True)(input)
                activation = copy.deepcopy(self.layersActivations[i])
                layer = activation(layer)
                if self.pooling[i]:
                    layer = MaxPool1D()(layer)
        if self.use_dropout:
            dropout = Dropout(self.dropout)(layer)
            layer = dropout
        if self.numOutputNeurons is not None:
            output = Dense(self.numOutputNeurons, activation=self.networkActivation,
                           kernel_regularizer=self.kernel_regularizer, bias_regularizer=self.bias_regularizer,
                           activity_regularizer=self.activity_regularizer)(layer)
        else:
            output = layer
        return input, output

    def create(self, model_summary_filename=None):
        input, output = self.build_network()
        model = Model(inputs=input, outputs=output)
        model.compile(loss=self.loss, optimizer=self.optimizer, metrics=self.metrics)
        if model_summary_filename is not None:
            with open(model_summary_filename, 'w') as summary_file:
                model.summary(print_fn=lambda x: summary_file.write(x + '\n'))
        return adapter.KerasAdapter(model)

    @staticmethod
    def create_from_path(filepath, custom_objects=None):
        model = load_model(filepath, custom_objects=custom_objects)
        return adapter.KerasAdapter(model)


class MixedInputModelCreator(ModelCreator):

    def __init__(self, input_models: [Model], outputUnits, numOutputNeurons,
                 layersActivations=None, networkActivation='sigmoid',
                 loss='categorical_crossentropy', optimizer='adam', use_dropout=False, dropout=0.5,
                 metrics=['accuracy'], kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None):
        self.input_models = input_models
        self.output_units = outputUnits
        self.num_output_neurons = numOutputNeurons
        self.network_activation = networkActivation
        self.layers_activation = layersActivations
        self.loss = loss
        self.optimizer = optimizer
        self.use_dropout = use_dropout
        self.dropout = dropout
        self.metrics = metrics
        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer
        self.activity_regularizer = activity_regularizer
        self.name = "MIXED_MODEL"

    def build_network(self):
        inputs = []
        outputs = []
        for model in self.input_models:
            inputs.append(model.input)
            outputs.append(model.output)
        input = concatenate(outputs)
        layer = Dense(self.output_units[0])(input)
        activation = copy.deepcopy(self.layers_activation[0])
        layer = activation(layer)
        if len(self.output_units) > 1:
            for i in range(1, len(self.output_units)):
                if self.use_dropout:
                    dropout = Dropout(self.dropout)(layer)
                    layer = dropout
                layer = Dense(self.output_units[i])(layer)
                activation = copy.deepcopy(self.layers_activation[i])
                layer = activation(layer)
        if self.use_dropout:
            dropout = Dropout(self.dropout)(layer)
            layer = dropout
        output = Dense(self.num_output_neurons, activation=self.network_activation,
                       kernel_regularizer=self.kernel_regularizer, bias_regularizer=self.bias_regularizer,
                       activity_regularizer=self.activity_regularizer)(layer)
        return inputs, output

    def create(self, model_summary_filename=None):
        input, output = self.build_network()
        model = Model(inputs=input, outputs=output)
        model.compile(loss=self.loss, optimizer=self.optimizer, metrics=self.metrics)
        plot_model(model, to_file="model.png")
        if model_summary_filename is not None:
            with open(model_summary_filename, 'w') as summary_file:
                model.summary(print_fn=lambda x: summary_file.write(x + '\n'))
        return adapter.KerasAdapter(model)

    @staticmethod
    def create_from_path(filepath, custom_objects=None):
        model = load_model(filepath, custom_objects=custom_objects)
        return adapter.KerasAdapter(model)


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


def repeat(x):

    stepMatrix = K.ones_like(x[0][:,:,:1]) #matrix with ones, shaped as (batch, steps, 1)
    latentMatrix = K.expand_dims(x[1],axis=1) #latent vars, shaped as (batch, 1, latent_dim)

    return K.batch_dot(stepMatrix,latentMatrix)

class KerasVariationalAutoencoder(ModelCreator):
    """
    A class that create a Variational Autoenconder.
    This script was writen based on https://github.com/keras-team/keras/blob/master/examples/variational_autoencoder.py
    See the linked script for more details on a tutorial of how to build it.
    """
    def __init__(self, input_shape, intermediate_dim, latent_dim, optmizer='adam', loss='mse'):
        self.input_shape = input_shape
        self.intermediate_dim = intermediate_dim
        self.latent_dim = latent_dim
        self.sampling = sampling
        self.loss = loss
        self.optmizer = optmizer

    def create(self):
        encoder, decoder, vae = self.__build_recurrent_model()
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

    def __build_recurrent_model(self):
        # Encoder
        inputs = Input(shape=self.input_shape, name='encoder_input')
        x = LSTM(self.intermediate_dim)(inputs)
        z_mean = Dense(self.latent_dim)(x)
        z_log_var = Dense(self.latent_dim)(x)
        # Z layer
        z = Lambda(self.sampling, name='z')([z_mean, z_log_var])

        print(self.input_shape)
        # Decoder
        latent_inputs = Lambda(repeat)([inputs,z])
        # latent_inputs = RepeatVector(self.input_shape[0])(z)
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


def generate_n_ints(n, min_int, max_int, int_step, name_pattern, hp):
    random_ints = []
    for i in range(n):
        generated_int = hp.Int(name_pattern.format(i),
               min_value=min_int,
               max_value=max_int,
               step=int_step)
        random_ints.append(generated_int)
    return random_ints


def generate_n_choices(n, choices, name_pattern, hp):
    random_choices = []
    for i in range(n):
        choice = hp.Choice(name_pattern.format(i), choices)
        random_choices.append(choice)
    return random_choices


def generate_n_int_arrays(n, min_size, max_size, size_step, array_name_pattern,
                          min_int, max_int, int_step, int_name_pattern, hp):
    random_arrays = []
    for i in range(n):
        array_size = hp.Int(array_name_pattern.format(i),
               min_value=min_size,
               max_value=max_size,
               step=size_step)
        random_array = generate_n_ints(array_size, min_int, max_int, int_step,
                                       array_name_pattern.format(i) + '_' + int_name_pattern, hp)
        random_arrays.append(random_array)
    return random_arrays


def get_activation_by_string(activation_string):
    if activation_string == "leakyrelu":
        return LeakyReLU()


def get_optimizers_by_string(optimizer_string):
    if optimizer_string == "adam":
        return Adam(
            learning_rate=0.01,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-07,
            amsgrad=True
        )


class MultilayerTemporalConvolutionalNNHyperModel(HyperModel):

    name = "TCN_MODEL"

    def __init__(self, input_shape, output_units, metrics, params,
                 model_input_name:str=None):
        self.input_shape = input_shape
        self.output_units = output_units
        self.params = params
        self.metrics = metrics
        self.model_input_name = model_input_name

    def build(self, hp):
        layers = hp.Int('layers',
                        min_value=self.params['layers_min'],
                        max_value=self.params['layers_max'],
                        step=self.params['layers_step'])
        print("Layers ", layers )
        hidden_output_units = generate_n_ints(layers, self.params['hidden_output_units_min'],
                                              self.params['hidden_output_units_max'],
                                              self.params['hidden_output_units_steps'],
                                              'units_{}', hp)
        print("Hidden units ", hidden_output_units)
        loss = hp.Choice('loss', self.params['losses'])
        print("Loss ", loss)
        hidden_activations = generate_n_choices(layers, self.params['hidden_activations'],'activation_{}', hp)
        aux = []
        for activation in hidden_activations:
            aux.append(get_activation_by_string(activation))
        hidden_activations = aux
        print("Activations ", hidden_activations)
        network_activation = hp.Choice('networkActivation', self.params['network_activations'])
        # network_activation = get_activation_by_string(network_activation)
        print("Net activations ", network_activation)
        pooling = generate_n_choices(layers, self.params['hidden_pooling'], 'pooling_{}', hp)
        print("Pooling ", pooling)
        kernel_sizes = generate_n_ints(layers, self.params['kernel_size_min'], self.params['kernel_size_max'],
                                       self.params['kernel_size_step'], 'kernel_size_{}', hp)
        print("Kernel sizes ", kernel_sizes)
        dilations = generate_n_int_arrays(layers, self.params['min_dilation_size'], self.params['max_dilation_size'],
                                          self.params['dilation_size_step'], 'dilation_{}', self.params['min_dilation'],
                                          self.params['max_dilation'], self.params['dilation_step'], 'num_{}', hp)
        print("Dilations ", dilations)
        num_stacks = generate_n_ints(layers, self.params['stacks_min'], self.params['stacks_max'],
                                     self.params['stacks_steps'], 'stack_{}', hp)
        print("Number of stacks ", num_stacks)
        use_dropout = hp.Choice('use_dropout', self.params['use_dropout'])
        print("Use dropout? ", use_dropout)
        dropout = None
        if use_dropout:
            dropout = hp.Choice('dropout', self.params['dropout_choices'])
        print("Dropout ", dropout)
        optimizer = hp.Choice('optimizer', self.params['optimizers'])
        optimizer = get_optimizers_by_string(optimizer)
        print("Network optimizer ", optimizer)
        model_creator = MultilayerTemporalConvolutionalNNCreator(self.input_shape,
                                                 hidden_output_units,
                                                 self.output_units,
                                                 loss=loss,
                                                 layersActivations=hidden_activations,
                                                 networkActivation=network_activation,
                                                 pooling=pooling,
                                                 kernel_sizes=kernel_sizes,
                                                 use_dropout=use_dropout,
                                                 dilations=dilations,
                                                 nb_stacks=num_stacks,
                                                 dropout=dropout,
                                                 kernel_regularizer=None,
                                                 metrics=self.metrics,
                                                 model_input_name=self.model_input_name,
                                                 optimizer=optimizer)
        adapter = model_creator.create()
        model = adapter.model
        return model


class MultilayerKerasRecurrentNNHyperModel(HyperModel):

    name = "RNN_MODEL"

    def __init__(self, input_shape, output_units, is_gru, metrics, params):
        self.input_shape = input_shape
        self.output_units = output_units
        self.is_gru = is_gru
        self.params = params
        self.metrics = metrics
        if is_gru:
            self.name = "GRU_MODEL"
        else:
            self.name = "LSTM_MODEL"

    def build(self, hp):
        layers = hp.Int('layers',
                        min_value=self.params['layers_min'],
                        max_value=self.params['layers_max'],
                        step=self.params['layers_step'])
        print("Layers ", layers )
        hidden_output_units = generate_n_ints(layers, self.params['hidden_output_units_min'],
                                              self.params['hidden_output_units_max'],
                                              self.params['hidden_output_units_steps'],
                                              'units_{}', hp)
        print("Hidden units ", hidden_output_units)
        loss = hp.Choice('loss', self.params['losses'])
        print("Loss ", loss)
        hidden_activations = generate_n_choices(layers, self.params['hidden_activations'],'activation_{}', hp)
        aux = []
        for activation in hidden_activations:
            aux.append(get_activation_by_string(activation))
        hidden_activations = aux
        print("Activations ", hidden_activations)
        network_activation = hp.Choice('networkActivation', self.params['network_activations'])
        # network_activation = get_activation_by_string(network_activation)
        print("Net activations ", network_activation)
        pooling = generate_n_choices(layers, self.params['hidden_pooling'], 'pooling_{}', hp)
        print("Pooling ", pooling)
        use_dropout = hp.Choice('use_dropout', self.params['use_dropout'])
        print("Use dropout? ", use_dropout)
        dropout = None
        if use_dropout:
            dropout = hp.Choice('dropout', self.params['dropout_choices'])
        print("Dropout ", dropout)
        optimizer = hp.Choice('optimizer', self.params['optimizers'])
        optimizer = get_optimizers_by_string(optimizer)
        print("Network optimizer ", optimizer)
        model_creator = MultilayerKerasRecurrentNNCreator(self.input_shape,
                                                        hidden_output_units,
                                                         self.output_units,
                                                         loss=loss,
                                                         layersActivations=hidden_activations,
                                                         networkActivation=network_activation,
                                                         gru=self.is_gru,
                                                          kernel_regularizer=None,
                                                         use_dropout=use_dropout,
                                                        dropout=dropout,
                                                         metrics=self.metrics,
                                                         optimizer=optimizer)
        adapter = model_creator.create()
        model = adapter.model
        return model