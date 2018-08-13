import itertools
import sys

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from utils.classification_report import classification_report
from utils.plot_confusion_matrix_util import plot_confusion_matrix


def evaluate(model, test, test_input, labels_vocab, save_path, name):
    test_eval = model.evaluate(test_input, np.array(test.y))
    print('Test loss:', test_eval[0])
    print('Test accuracy:', test_eval[1])

    predicted_values = np.argmax(model.predict(test_input), axis=-1)
    true_values = np.argmax(test.y, -1)

    # flatten to single array with class labels
    true_values = list(itertools.chain(*true_values))
    predicted_values = list(itertools.chain(*predicted_values))

    orig_stdout = sys.stdout
    f = open(save_path + 'results.txt', 'w')
    sys.stdout = f

    print("Macro Precision/Recall/F1 score:")
    print(precision_recall_fscore_support(true_values, predicted_values, average='macro'))
    print(60 * "-")

    print("Micro Precision/Recall/F1 score:")
    print(precision_recall_fscore_support(true_values, predicted_values, average='micro'))
    print(60 * "-")

    keys = list(labels_vocab.stoi.keys())
    values = list(labels_vocab.stoi.values())

    # Classification report's
    macro_report = classification_report(true_values, predicted_values, labels=values, target_names=keys, digits=4, average='macro')
    print(macro_report)
    print(60 * "-")

    micro_report = classification_report(true_values, predicted_values, labels=values, target_names=keys, digits=4, average='micro')
    print(micro_report)

    sys.stdout = orig_stdout
    f.close()

    # Confusion Matrix
    cnf_matrix = confusion_matrix(true_values, predicted_values)
    np.set_printoptions(precision=2)
    plot_confusion_matrix(cnf_matrix, classes=list(labels_vocab.stoi.keys()), normalize=True, title='Confusion matrix - ' + name)
    plt.savefig(save_path + '/images/confusion_matrix.png', dpi=200, format='png', bbox_inches='tight')
    plt.close()
