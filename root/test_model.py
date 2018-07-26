import itertools

import matplotlib.pyplot as plt
import numpy as np
from keras.models import load_model
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from root.constants import MAX_LEN
from utils.classification_report import classification_report
from utils.plot_confusion_matrix_util import plot_confusion_matrix


def test_model(test, text_vocab, labels_vocab):
    model = load_model('../models/ner_model')

    predicted_values = np.argmax(model.predict([test.X, test.pos, np.array(test.characters).reshape((len(test.characters), MAX_LEN, 10))]),
                                 axis=-1)
    true_values = np.argmax(test.y, -1)

    # flatten to single array with class labels
    true_values = list(itertools.chain(*true_values))
    predicted_values = list(itertools.chain(*predicted_values))

    print("Macro Precision/Recall/F1 score:")
    print(precision_recall_fscore_support(true_values, predicted_values, average='macro'))

    print("Micro Precision/Recall/F1 score:")
    print(precision_recall_fscore_support(true_values, predicted_values, average='micro'))

    print("Weighted Precision/Recall/F1 score:")
    print(precision_recall_fscore_support(true_values, predicted_values, average='weighted'))

    # Remove padding label
    keys = list(labels_vocab.stoi.keys())
    values = list(labels_vocab.stoi.values())
    # values.remove(labels_vocab.stoi[NO_ENTITY_TOKEN])
    # keys.remove(NO_ENTITY_TOKEN)

    # Classification report
    report = classification_report(true_values, predicted_values, labels=values, target_names=keys, digits=4, average='macro')
    print(report)

    # plot_classification_report(report)
    # plt.savefig('../results/classification_report.png', dpi=200, format='png', bbox_inches='tight')
    # plt.close()

    # Confusion Matrix
    cnf_matrix = confusion_matrix(true_values, predicted_values)
    np.set_printoptions(precision=2)
    # TODO fix classes
    plot_confusion_matrix(cnf_matrix, classes=list(labels_vocab.stoi.keys()), normalize=True, title='Normalized confusion matrix')
    plt.savefig('../results/confusion_matrix.png', dpi=200, format='png', bbox_inches='tight')
    plt.close()
