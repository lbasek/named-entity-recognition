import itertools

import numpy as np
from keras.models import load_model
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

from root.constants import NO_ENTITY_TOKEN
from utils.plot_classification_report_util import plot_classification_report


def test_model(test, text_vocab, labels_vocab):
    model = load_model('../models/ner_model')

    predicted_values = np.argmax(model.predict([test.X, test.pos]), axis=-1)
    true_values = np.argmax(test.y, -1)

    # flatten to single array with class labels
    true_values = list(itertools.chain(*true_values))
    predicted_values = list(itertools.chain(*predicted_values))

    # Remove padding label
    keys = list(labels_vocab.stoi.keys())
    values = list(labels_vocab.stoi.values())

    # values.remove(labels_vocab.stoi[NO_ENTITY_TOKEN])
    # keys.remove(NO_ENTITY_TOKEN)

    report = classification_report(true_values, predicted_values, labels=values, target_names=keys)
    print(report)

    plot_classification_report(report)
    plt.savefig('../results/classification_report.png', dpi=200, format='png', bbox_inches='tight')
    plt.close()
