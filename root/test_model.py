import itertools

from nltk.tokenize import word_tokenize
import numpy as np
from keras.models import load_model
from keras_preprocessing.sequence import pad_sequences
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

from root.constants import NO_ENTITY_TOKEN, MAX_LEN, PAD
from utils.plot_classification_report_util import plot_classification_report


def test_model(test, text_vocab, labels_vocab, pos=False):
    model = load_model('../models/ner_model')

    true_values = np.argmax(test.y, -1)
    if pos:
        predicted_values = np.argmax(model.predict([test.X, test.pos]), axis=-1)
    else:
        predicted_values = np.argmax(model.predict(test.X), axis=-1)

    # flatten to single array with class labels
    true_values = list(itertools.chain(*true_values))
    predicted_values = list(itertools.chain(*predicted_values))

    # Remove padding label
    keys = list(labels_vocab.stoi.keys())
    values = list(labels_vocab.stoi.values())

    # values.remove(labels_vocab.stoi[PAD])
    # keys.remove(PAD)

    report = classification_report(true_values, predicted_values, labels=values, target_names=keys)
    print(report)

    plot_classification_report(report)
    plt.savefig('../results/classification_report.png', dpi=200, format='png', bbox_inches='tight')
    plt.close()

    print("{:15}||{:5}||{}".format("Word", "True", "Pred"))
    print(30 * "=")

    flat_test_x = list(itertools.chain(*test.X))

    test_sentence = word_tokenize(
        "My name is John and I live in New York. I am dutch and I working in Ajax Amsterdam.")

    x_test_sent = pad_sequences(sequences=[[text_vocab.stoi[w] for w in test_sentence]],
                                padding="post", value=0, maxlen=MAX_LEN)

    if pos:
        p = model.predict([x_test_sent, test.pos])
    else:
        p = model.predict(np.array([x_test_sent[0]]))

    p = np.argmax(p, axis=-1)
    print("{:15}||{}".format("Word", "Prediction"))
    print(30 * "=")
    for w, pred in zip(test_sentence, p[0]):
        print("{:15}: {:5}".format(w, labels_vocab.itos[pred]))

    # exit(1)
    #
    # for i in range(0, 5000):
    #     # if text_vocab.itos[flat_test_x[i]] != PAD:
    #     print("{:15}: {:5} {}".format(text_vocab.itos[flat_test_x[i]], labels_vocab.itos[true_values[i]],
    #                                   labels_vocab.itos[predicted_values[i]]))
