import abc

from keras.engine.saving import load_model
from keras.layers.convolutional import Conv2D, Conv1D
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.pooling import MaxPool2D, MaxPool1D, AveragePooling1D
from keras.layers.recurrent import LSTM, GRU
from keras.models import Sequential
from keras.optimizers import SGD
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

class KerasCovolutionalNNCreator(ModelCreator):

    #TODO: Finish this class
    def __init__(self, input_shape=None, num_class=8, loss='mse', optimizer='sgd'):
        self.input_shape = input_shape
        self.loss = loss
        self.optimizer = optimizer
        self.num_class=num_class

    def __build_model(self):
        model = Sequential()
        model.add(Conv1D(256, kernel_size=8, activation='relu', padding='same', input_shape=self.input_shape))
        model.add(AveragePooling1D(pool_size=1, padding="same"))
        # model.add(Dropout(0.5))
        # model.add(Conv1D(128, kernel_size=3, activation='elu', padding='same'))
        # model.add(AveragePooling1D(pool_size=1, padding="same"))
        # model.add(Dropout(0.5))
        # model.add(Conv1D(16, kernel_size=5, activation='elu', padding='same'))
        # model.add(MaxPool1D(pool_size=1, padding="same"))
        # model.add(Conv1D(16, kernel_size=5, activation='elu', padding='same'))
        # model.add(Dropout(0.25))

        # model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(128))
        model.add(Dropout(0.5))
        model.add(Dense(self.num_class, activation='tanh'))
        # sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(loss=self.loss, optimizer=self.optimizer, metrics=['accuracy'])
        return model

    def create(self):
        return adapter.KerasGeneratorAdapter(self.__build_model())

class KerasMultilayerPerceptron(ModelCreator):

    def __init__(self, num_class=8, input_dim=200, layers=1, hidden_units=[20], use_dropout=True, dropout=0.5):
        if len(hidden_units) != layers:
            raise ValueError("The hidden_units must have the size of the number of layers.")
        self.input_dim = input_dim
        self.layers = layers
        self.hidden_units = hidden_units
        self.use_dropout = use_dropout
        self.dropout = dropout
        self.num_class = num_class

    def __build_model(self):
        model = Sequential()
        i = 0
        for i in range(self.layers):
            model.add(Dense(self.hidden_units[i], activation='relu', input_dim=self.input_dim))
            if self.use_dropout:
                model.add(Dropout(self.dropout))
        model.add(Dense(self.num_class, activation='softmax'))
        sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(loss='categorical_crossentropy',
                      optimizer=sgd,
                      metrics=['accuracy'])
        return model

    def create(self):
        return adapter.KerasGeneratorAdapter(self.__build_model())

# class SimpleKerasCovolutionalNNCreator(ModelCreator):
#
#     def __build_model(self):
#
#
#     def create(self):

class SklearnNeuralNetwork(ModelCreator):

    def __init__(self, solver="lbfgs", alpha=1e-5, hidden_layer_sizes=10, random_state=1):
        self.solver = solver
        self.alpha = alpha
        self.hidden_layer_sizes = hidden_layer_sizes
        self.random_state = random_state

    def create(self):
        model = MLPClassifier(solver=self.solver, alpha=self.alpha, hidden_layer_sizes=self.hidden_layer_sizes, random_state=self.random_state)
        return adapter.SklearnAdapter(model)
