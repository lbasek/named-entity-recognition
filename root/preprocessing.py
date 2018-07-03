import itertools
import pandas as pd
from keras.utils import to_categorical
from keras_preprocessing.sequence import pad_sequences
from enum import Enum
import matplotlib.pyplot as plt
from utils.sentence_getter import SentenceGetter


class Preprocessing(object):

    def __init__(self, train_filename, test_filename, validation_filename):
        self.train_filename = train_filename
        self.test_filename = test_filename
        self.validation_filename = validation_filename
        # TODO maxlen
        self.MAX_LEN = 120

    def create_input(self, dataset):

        input_dataset = self.input_dataset(dataset)

        data = pd.read_csv(input_dataset, encoding='utf-8')

        sentence_getter = SentenceGetter(data=data)

        sentences = sentence_getter.sentences

        plt.hist([len(s) for s in sentences], bins=50)
        plt.show()

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
        words = set(itertools.chain(*X))
        labels = set(itertools.chain(*Y))

        index2word = dict((i + 1, v) for i, v in enumerate(words))
        word2index = dict((v, i + 1) for i, v in enumerate(words))

        index2labels = dict((i + 1, v) for i, v in enumerate(labels))
        labels2index = dict((v, i + 1) for i, v in enumerate(labels))

        # # testing dictionary
        # tmp_index = word2index["Soccer"]
        # print(tmp_index)
        #
        # tmp_word = index2word[inx]
        # print(tmp_word)

        num_entities = len(labels2index) + 1
        num_words = len(word2index) + 1

        print('num_words = {0}, num_entities = {1}'.format(num_words, num_entities))

        X_enc = list(map(lambda x: [word2index[wx] for wx in x], X))
        Y_enc = list(map(lambda y: [labels2index[wy] for wy in y], Y))

        # print(Y_enc[1])
        # print(to_categorical(Y_enc[1], num_classes=num_entities))

        Y_one_hot_encode = list(map(lambda item: to_categorical(item, num_classes=num_entities), Y_enc))

        X_all = pad_sequences(sequences=X_enc, maxlen=self.MAX_LEN)
        Y_all = pad_sequences(sequences=Y_one_hot_encode, maxlen=self.MAX_LEN)

        return X_all, Y_all, num_entities, num_words

    def input_dataset(self, argument):
        switcher = {
            Dataset.train: self.train(),
            Dataset.test: self.test(),
            Dataset.validation: self.validation()
        }
        return switcher.get(argument, "NULL")

    def train(self):
        return self.train_filename

    def test(self):
        return self.test_filename

    def validation(self):
        return self.validation_filename


class Dataset(Enum):
    train = 1
    test = 2
    validation = 3
