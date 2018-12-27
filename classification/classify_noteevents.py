import json
import os

import keras
from keras import backend as K
import gensim
import math
from random import shuffle

import numpy
from nltk import regexp_tokenize
from sklearn.metrics.classification import f1_score
from sklearn.model_selection import train_test_split, KFold

from data_representation import Word2VecEmbeddingCreator
from model_creators import MultilayerKerasRecurrentNNCreator

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

def data_transform(data, word2vecModel, embeddingSize):
    embeddingCreator = Word2VecEmbeddingCreator(word2vecModel, embeddingSize)
    embeddingData = []
    for text in data:
        embeddingData.append(embeddingCreator.create_embedding_matrix(text))
    return embeddingData


sepsisFilesPath = "/home/mattyws/Documents/no_sepsis_noteevents"
noSepsisFilesPath = "/home/mattyws/Documents/sepsis_noteevents"

embeddingSize = 200

# Get files paths
sepsisFiles = []
for dir, path, files in os.walk(sepsisFilesPath):
    for file in files:
        sepsisFiles.append(dir + "/" + file)

noSepsisFiles = []
for dir, path, files in os.walk(noSepsisFilesPath):
    for file in files:
        noSepsisFiles.append(dir + "/" + file)

# Generating objects, this objects will be shuffle in the vector and then they will be separated into class and data vectors
noteeventsObjects = []
for filePath in sepsisFiles:
    with open(filePath) as file_handler:
        jsonObject = json.load(file_handler)
        texts = []
        for object in jsonObject:
            texts.append(object['text'])
        noteeventsObjects.append({'texts':texts, 'class':[1]})

lenSepsisObjects = len(noteeventsObjects)
for filePath in noSepsisFiles:
    if len(noteeventsObjects) - lenSepsisObjects >= math.ceil(lenSepsisObjects * 1.5) :
        break
    with open(filePath) as file_handler:
        jsonObject = json.load(file_handler)
        texts = []
        for object in jsonObject:
            texts.append(object['text'])
        noteeventsObjects.append({'texts': texts, 'class': [0]})

shuffle(noteeventsObjects)
# Separating data and class from the noteevents_objects

data = []
labels = []
for noteevent in noteeventsObjects:
    data.append('\n'.join(noteevent['texts']))
    labels.append(noteevent['class'])

dataTrain, data, labelsTrain, labels = train_test_split(data, labels, stratify=labels, test_size=0.4)

#TODO: proper preprocess the data
data = numpy.array(data)
labels = numpy.array(labels)

new_data = []
for d in data:
    new_data.append(regexp_tokenize(d.lower(), pattern='\w+|\$[\d\.]+|\S+'))

kf = KFold(n_splits=5, random_state=15)

# Network Parameters
inputShape = (None, embeddingSize)
outputUnits = [128, 64]
numOutputNeurons = 1
loss = 'binary_crossentropy'
layersActivations = ['relu', 'relu']
gru=True
useDropout=True
dropout=0.3
trainingEpochs = 100

word2vecWindow = 4
wordd2vecIter = 150

i = 1
f1_folds = []
for trainIndex, testIndex in kf.split(data):

    # Training an instance of Word2Vec model with the training data
    print("======== Fold {} ========".format(i))
    i += 1
    print("Training word2vec")
    word2vecModel = gensim.models.Word2Vec(data[trainIndex], size = embeddingSize, min_count=1, window=word2vecWindow,
                                           iter=wordd2vecIter, sg=1)
    dataTrain = data_transform(data[trainIndex], word2vecModel, embeddingSize)
    dataTest = data_transform(data[testIndex], word2vecModel, embeddingSize)
    modelCreator = MultilayerKerasRecurrentNNCreator(inputShape, outputUnits, numOutputNeurons,
                                                     loss=loss, layersActivations=layersActivations, gru=gru,
                                                     use_dropout=useDropout, dropout=dropout,
                                                     metrics=[f1, keras.metrics.binary_accuracy])
    kerasAdapter = modelCreator.create()
    kerasAdapter.fit(dataTrain, labels[trainIndex], epochs=trainingEpochs, batch_size=len(dataTrain), validationDocs=dataTest,
                     validationLabels=labels[testIndex], validationSteps=len(dataTest))
    #TODO: test the recurrent model created
    #TODO: measure the model's performance
    predicted = kerasAdapter.predict(dataTest, batch_size=len(dataTest))
    # passing to an list representation to use the sklearn f1_score function
    predicted_classes = []
    for p in predicted:
        predicted_classes.append(p[0])
    real_classes = []
    for l in labels[testIndex]:
        real_classes.append(l[0])

    f1_folds.append(f1_score(real_classes, predicted_classes))

print("Folds mean: {}".format( sum(f1_folds) / len(f1_folds) ))

