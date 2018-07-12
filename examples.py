import numpy as np
import gensim

from sif_embed import SIFEmbeddingVectorizer

def main():
    model = gensim.models.KeyedVectors.load_word2vec_format('data/glove.6B.50d_gensim.txt', binary=False)
    # each line is a word and its frequency
    weightfile = 'auxiliary_data/enwiki_vocab_min200.txt'

    transformer = SIFEmbeddingVectorizer(model, weightfile)

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
