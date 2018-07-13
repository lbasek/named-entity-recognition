import itertools

import numpy as np
from keras.models import load_model
from sklearn.metrics import classification_report
from root.constants import UNK_LBL, NO_ENTITY_TOKEN
import matplotlib.pyplot as plt

from utils.plot_classification_report_util import plot_classification_report


def test_model(X_test, y_test, text_vocab, labels_vocab):
    model = load_model('../models/ner_model')
    i = 200
    p = model.predict(np.array([X_test[i]]))
    p = np.argmax(p, axis=-1)
    true = np.argmax(y_test[i], -1)
    print("{:15}||{:5}||{}".format("Word", "True", "Pred"))
    print(30 * "=")
    for w, t, pred in zip(X_test[i], true, p[0]):
        if w != 0:
            print("{:15}: {:5} {}".format(text_vocab.itos[w], labels_vocab.itos[t], labels_vocab.itos[pred]))

    predicted_values = model.predict_classes(X_test)
    true_values = np.argmax(y_test, -1)

    # flatten to single array with class labels
    true_values = list(itertools.chain(*true_values))
    predicted_values = list(itertools.chain(*predicted_values))

    # Remove padding label
    keys = list(labels_vocab.stoi.keys())
    values = list(labels_vocab.stoi.values())
    values.remove(labels_vocab.stoi[UNK_LBL])
    keys.remove(UNK_LBL)
    values.remove(labels_vocab.stoi[NO_ENTITY_TOKEN])
    keys.remove(NO_ENTITY_TOKEN)

    report = classification_report(true_values, predicted_values, labels=values, target_names=keys)
    print(report)

    plot_classification_report(report)
    plt.savefig('classification_report.png', dpi=200, format='png', bbox_inches='tight')
    plt.close()
