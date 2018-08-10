import itertools
from collections import namedtuple

import numpy as np
from keras.utils import to_categorical
from keras_preprocessing.sequence import pad_sequences
from root.constants import NO_ENTITY_TOKEN, MAX_LEN, PAD, MAX_LEN_CHAR
from .data_processor import numericalize
from .vocab import TextVocab, LabelVocab, PosVocab, CharacterVocab


def load_dataset():
    # load examples
    train_examples = load_examples('../dataset/raw/train.txt')
    val_examples = load_examples('../dataset/raw/valid.txt')
    test_examples = load_examples('../dataset/raw/test.txt')

    # build vocabularies
    text_vocab = TextVocab.build(list(map(lambda e: e.sentence, train_examples)))
    labels_vocab = LabelVocab.build(list(map(lambda e: e.labels, train_examples)))
    pos_vocab = PosVocab.build(list(map(lambda e: e.pos, train_examples)))
    character_vocab = CharacterVocab.build([w_i for w in map(lambda e: e.sentence, train_examples) for w_i in w])

    train_set = create_dataset(train_examples, text_vocab, labels_vocab, pos_vocab, character_vocab)
    val_set = create_dataset(val_examples, text_vocab, labels_vocab, pos_vocab, character_vocab)
    test_set = create_dataset(test_examples, text_vocab, labels_vocab, pos_vocab, character_vocab)

    return text_vocab, labels_vocab, pos_vocab, character_vocab, train_set, val_set, test_set


def load_examples(file_path):
    """
    Loads sentences from file in CoNLL 2003 format.

    :param file_path: Path to file with CoNLL data.
    :return: list(Example)
    """
    examples = []
    sentence = []
    labels = []
    pos = []

    with open(file_path) as fd:
        for line in fd:
            line = line.rstrip('\n')
            if line.startswith('-DOCSTART-'):
                continue
            if not line:
                if sentence and labels and pos:
                    examples.append(Example(sentence, labels, pos))
                    sentence = []
                    labels = []
                    pos = []
                continue

            parts = line.split(' ')

            sentence.append(parts[0])

            if parts[1] in ['$', '"', '(', ')', "''", '.', ':', ',']:
                pos.append('NN')
            else:
                pos.append(parts[1])

            labels.append(parts[3])

    return examples


def one_hot_encode(matrix, num_classes):
    one_hot = [to_categorical(row, num_classes=num_classes) for row in matrix]
    return one_hot


def create_dataset(examples, text_vocab, labels_vocab, pos_vocab, character_vocab):
    X = numericalize(text_vocab, map(lambda e: e.sentence, examples), pad_token=PAD, maxlen=MAX_LEN)

    y = numericalize(labels_vocab, map(lambda e: e.labels, examples), pad_token=NO_ENTITY_TOKEN, maxlen=MAX_LEN)
    y = one_hot_encode(y.tolist(), len(labels_vocab.itos))

    pos = numericalize(pos_vocab, map(lambda e: e.pos, examples), pad_token=PAD, maxlen=MAX_LEN)

    characters = []
    for sentence in map(lambda e: e.sentence, examples):
        sent_seq = []
        for i in range(MAX_LEN):
            word_seq = []
            for j in range(MAX_LEN_CHAR):
                try:
                    word_seq.append(character_vocab.stoi[sentence[i][0][j]])
                except:
                    word_seq.append(character_vocab.stoi[PAD])
            sent_seq.append(word_seq)
        characters.append(np.array(sent_seq))

    return Dataset(X, y, pos, characters)


Example = namedtuple('Example', 'sentence labels pos')
Dataset = namedtuple('Dataset', 'X y pos characters')
