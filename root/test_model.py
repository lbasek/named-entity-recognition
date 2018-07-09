import numpy as np
from keras.models import load_model


def test_model(X_test, y_test, index2word, index2label):
    model = load_model('../models/ner_model')
    i = 320
    p = model.predict(np.array([X_test[i]]))
    p = np.argmax(p, axis=-1)
    true = np.argmax(y_test[i], -1)
    print("{:15}||{:5}||{}".format("Word", "True", "Pred"))
    print(30 * "=")
    for w, t, pred in zip(X_test[i], true, p[0]):
        if w != 0 or index2word[w] != '<END-PAD>':
            print("{:15}: {:5} {}".format(index2word[w], index2label[t], index2label[pred]))
