import sys
import numpy as np

GLOVE_DIR = '../embedding/glove.6B.100d.txt'


def get_pretrained_glove(num_words, text_vocab):
    embeddings_index = {}
    try:
        f = open(GLOVE_DIR, 'r+', encoding="utf-8")
    except IOError:
        print("Can't open Glove file.")
        sys.exit(0)

    for line in f:
        values = line.split()
        word = values[0]
        coefficients = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefficients

    f.close()

    embedding_matrix = np.zeros((num_words, 100))

    for index, word in enumerate(text_vocab.itos):
        embedding_vector = embeddings_index.get(word.lower())
        if embedding_vector is not None:
            # words not found in embedding index will be zero
            embedding_matrix[index] = embedding_vector
