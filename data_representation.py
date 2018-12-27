import numpy as np


class Word2VecEmbeddingCreator(object):

    """
    A class that transforms a text into their representation of word embedding
    It uses a trained word2vec model model to build a 3 dimentional vector representation of the document.
     The first dimension represents the document, the second dimension represents the word and the third dimension is the word embedding array
    """

    def __init__(self, word2vecModel, embeddingSize=200):
        self.word2vecModel = word2vecModel
        self.embeddingSize = embeddingSize
        self.num_docs = 0

    def create_embedding_matrix(self, text, max_words=None):
        """
        Transform a tokenized text into a 2 dimensional array with the word2vec model
        :param text: the tokenized text
        :param max_words: the max number of words to put into the 3 dimensional array
        :return: the 3 dimensional array representing the content of the tokenized text
        """
        if max_words is None:
            x = np.zeros(shape=(1, len(text), self.embeddingSize), dtype='float')
        else:
            x = np.zeros(shape=(1, max_words, self.embeddingSize), dtype='float')
        for pos, w in enumerate(text):
            if max_words is not None and pos >= max_words:
                break
            try:
                x[0, pos] = self.word2vecModel[w]
            except:
                x[0, pos] = np.zeros(shape=self.embeddingSize)
        return x
