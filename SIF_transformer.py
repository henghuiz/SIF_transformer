from sklearn.base import BaseEstimator
import numpy as np
from sklearn.decomposition import TruncatedSVD


def get_weighted_average(embedding, x, w):
    """
    Compute the weighted average vectors
    :param embedding: embedding[i,:] is the vector for word i
    :param x: x[i, :] are the indices of the words in sentence i
    :param w: w[i, :] are the weights for the words in sentence i
    :return: emb[i, :] are the weighted average vector for sentence i
    """
    n_samples = x.shape[0]
    emb = np.zeros((n_samples, embedding.shape[1]))
    for i in range(n_samples):
        emb[i, :] = w[i, :].dot(embedding[x[i, :], :]) / np.count_nonzero(w[i, :])
    return emb


def compute_pc(x, npc=1):
    """
    Compute the principal components. DO NOT MAKE THE DATA ZERO MEAN!
    :param x: X[i,:] is a data point
    :param npc: number of principal components to remove
    :return: component_[i,:] is the i-th pc
    """
    svd = TruncatedSVD(n_components=npc, n_iter=7, random_state=0)
    svd.fit(x)
    return svd.components_


def remove_pc(x, npc=1):
    """
    Remove the projection on the principal components
    :param x: X[i,:] is a data point
    :param npc: number of principal components to remove
    :return: XX[i, :] is the data point after removing its projection
    """
    pc = compute_pc(x, npc)
    if npc == 1:
        XX = x - x.dot(pc.transpose()) * pc
    else:
        XX = x - x.dot(pc.transpose()).dot(pc)
    return XX


def SIF_embedding(embedding, x, w, rmpc):
    """
    Compute the scores between pairs of sentences using weighted average + removing the projection on the first principal component
    :param embedding: embedding[i,:] is the vector for word i
    :param x: x[i, :] are the indices of the words in the i-th sentence
    :param w: w[i, :] are the weights for the words in the i-th sentence
    :return: emb, emb[i, :] is the embedding for sentence i
    """
    emb = get_weighted_average(embedding, x, w)
    if rmpc > 0:
        emb = remove_pc(emb, rmpc)
    return emb


def load_word2vec(textfile):
    words = {}
    We = []
    f = open(textfile, 'r')
    lines = f.readlines()
    for (n, i) in enumerate(lines):
        i = i.split()
        j = 1
        v = []
        while j < len(i):
            v.append(float(i[j]))
            j += 1
        words[i[0]] = n
        We.append(v)
    return (words, np.array(We))


def prepare_data(list_of_seqs):
    lengths = [len(s) for s in list_of_seqs]
    n_samples = len(list_of_seqs)
    maxlen = np.max(lengths)
    x = np.zeros((n_samples, maxlen)).astype('int32')
    x_mask = np.zeros((n_samples, maxlen)).astype('float32')
    for idx, s in enumerate(list_of_seqs):
        x[idx, :lengths[idx]] = s
        x_mask[idx, :lengths[idx]] = 1.
    x_mask = np.asarray(x_mask, dtype='float32')
    return x, x_mask


def lookupIDX(words, w):
    w = w.lower()
    if len(w) > 1 and w[0] == '#':
        w = w.replace("#", "")
    if w in words:
        return words[w]
    elif 'UUUNKKK' in words:
        return words['UUUNKKK']
    else:
        return len(words) - 1


def get_sequences(p1, words):
    p1 = p1.split()
    X1 = []
    for i in p1:
        X1.append(lookupIDX(words, i))
    return X1


def sentences2idx(sentences, words):
    """
    Given a list of sentences, output array of word indices that can be fed into the algorithms.
    :param sentences: a list of sentences
    :param words: a dictionary, words['str'] is the indices of the word 'str'
    :return: x1, m1. x1[i, :] is the word indices in sentence i, m1[i,:] is the mask for sentence i (0 means no word at the location)
    """
    seq = []
    for i in sentences:
        seq.append(get_sequences(i, words))
    x1, m1 = prepare_data(seq)
    return x1, m1


def get_word_weight(weightfile, a=1e-3):
    if a <= 0:  # when the parameter makes no sense, use unweighted
        a = 1.0

    word2weight = {}
    with open(weightfile) as f:
        lines = f.readlines()
    N = 0
    for i in lines:
        i = i.strip()
        if (len(i) > 0):
            i = i.split()
            if (len(i) == 2):
                word2weight[i[0]] = float(i[1])
                N += float(i[1])
            else:
                print(i)
    for key, value in list(word2weight.items()):
        word2weight[key] = a / (a + value / N)
    return word2weight


def get_weight(words, word2weight):
    weight4ind = {}
    for word, ind in list(words.items()):
        if word in word2weight:
            weight4ind[ind] = word2weight[word]
        else:
            weight4ind[ind] = 1.0
    return weight4ind


def seq2weight(seq, mask, weight4ind):
    weight = np.zeros(seq.shape).astype('float32')
    for i in range(seq.shape[0]):
        for j in range(seq.shape[1]):
            if mask[i, j] > 0 and seq[i, j] >= 0:
                weight[i, j] = weight4ind[seq[i, j]]
    weight = np.asarray(weight, dtype='float32')
    return weight


class SIFEmbeddingVectorizer(BaseEstimator):
    def __init__(self, word2vec, word_frequency, weightpara=1e-3, rmpc=1):
        self.word2vec = word2vec
        self.word_frequency = word_frequency
        self.weightpara = weightpara
        self.rmpc = rmpc

        (self.words, self.We) = load_word2vec(word2vec)
        word2weight = get_word_weight(word_frequency, weightpara)
        self.weight4ind = get_weight(self.words, word2weight)

    def fit(self, X, y):
        return self

    def transform(self, X):
        x, m = sentences2idx(X, self.words)
        # x is the array of word indices, m is the binary mask indicating whether there is a word in that location
        w = seq2weight(x, m, self.weight4ind)  # get word weights

        # get SIF embedding
        embedding = SIF_embedding(self.We, x, w, self.rmpc)  # embedding[i,:] is the embedding for sentence i

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
