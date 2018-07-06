from sklearn.base import BaseEstimator
import numpy as np
import SIF


class SIFEmbeddingVectorizer(BaseEstimator):
    def __init__(self, word2vec, word_frequency, weightpara=1e-3, rmpc=1):
        self.word2vec = word2vec
        self.word_frequency = word_frequency
        self.weightpara = weightpara
        self.rmpc = rmpc

        (self.words, self.We) = SIF.getWordmap(word2vec)
        word2weight = SIF.getWordWeight(word_frequency, weightpara)
        self.weight4ind = SIF.getWeight(self.words, word2weight)

    def fit(self, X, y):
        return self

    def transform(self, X):
        x, m = SIF.sentences2idx(X, self.words)
        # x is the array of word indices, m is the binary mask indicating whether there is a word in that location
        w = SIF.seq2weight(x, m, self.weight4ind)  # get word weights

        # set parameters
        params = SIF.SIFParameter()
        params.rmpc = self.rmpc
        # get SIF embedding
        embedding = SIF.SIF_embedding(self.We, x, w, params)  # embedding[i,:] is the embedding for sentence i

        return embedding


def main():
    wordfile = 'data/glove.6B.50d.txt'
    weightfile = 'auxiliary_data/enwiki_vocab_min200.txt'  # each line is a word and its frequency

    transformer = SIFEmbeddingVectorizer(wordfile, weightfile)

    sentences = ['this is an example sentence', 'this is another sentence that is slightly longer',
                 'the quick brown fox jumps over the lazy dog']

    embedding = transformer.transform(sentences)

    emb1 = embedding[0, :]
    emb2 = embedding[1, :]
    inn = (emb1 * emb2).sum()
    emb1norm = np.sqrt((emb1 * emb1).sum())
    emb2norm = np.sqrt((emb2 * emb2).sum())
    score = inn / emb1norm / emb2norm

    print(score)


if __name__ == '__main__':
    main()
