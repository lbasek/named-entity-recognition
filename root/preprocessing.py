import itertools
import pandas as pd
from keras.utils import to_categorical
from keras_preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt
from utils.sentence_getter import SentenceGetter

TRAIN = '../dataset/csv/train.csv'
TEST = '../dataset/csv/test.csv'
VALIDATION = '../dataset/csv/valid.csv'


class Preprocessing(object):

    def __init__(self):
        self.MAX_LEN = 120

    def create_input(self):

        train = pd.read_csv(TRAIN, encoding='utf-8')
        valid = pd.read_csv(VALIDATION, encoding='utf-8')
        test = pd.read_csv(TEST, encoding='utf-8')

        frames = [train, valid, test]
        result = pd.concat(frames)

        sentence_getter = SentenceGetter(data=result)

        sentences = sentence_getter.sentences

        # plt.hist([len(s) for s in sentences], bins=50)
        # plt.show()

        X, Y = [], []

        for i in sentences:
            x, y = [], []
            for n in sentence_getter.get_next():
                x.append(n[0])
                y.append(n[3])

            if (len(x) > self.MAX_LEN) or (len(x) <= 1):
                continue

            X.append(x)
            Y.append(y)

        # build vocabulary of words and entities
        words = list(set(itertools.chain(*X)))
        words.append("<END-PAD>")
        labels = list(set(itertools.chain(*Y)))

        word2index = {w: i for i, w in enumerate(words)}
        labels2index = {t: i for i, t in enumerate(labels)}

        index2word = {i: w for i, w in enumerate(words)}
        index2labels = {i: t for i, t in enumerate(labels)}

        num_entities = len(labels2index)
        num_words = len(word2index)

        print('num_words = {0}, num_entities = {1}'.format(num_words, num_entities))

        X_enc = [[word2index[wrd] for wrd in sentence] for sentence in X]
        X_pad = pad_sequences(maxlen=self.MAX_LEN, sequences=X_enc, padding="post", value=word2index['<END-PAD>'])

        Y_enc = [[labels2index[lbl] for lbl in sentence] for sentence in Y]
        Y_pad = pad_sequences(maxlen=self.MAX_LEN, sequences=Y_enc, padding="post", value=labels2index['O'])

        Y_one_hot_enc = list(map(lambda item: to_categorical(item, num_classes=num_entities), Y_pad))

        return X_pad, Y_one_hot_enc, num_entities, num_words