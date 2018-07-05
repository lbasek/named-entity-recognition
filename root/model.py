import numpy as np
from datetime import datetime
from keras import Sequential
from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional, Activation
from keras.utils.vis_utils import plot_model
from tensorflow import keras
from keras.callbacks import TensorBoard


class NeuralNetwork(object):

    def __init__(self, num_words, num_entities, X_train, Y_train, X_validation, Y_validation, X_test, Y_test):
        self.num_words = num_words
        self.num_entities = num_entities
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_validation = X_validation
        self.Y_validation = Y_validation
        self.X_test = X_test
        self.Y_test = Y_test

    def train(self):
        model = Sequential()
        model.add(Embedding(input_dim=self.num_words, output_dim=120, input_length=120))
        model.add(Dropout(0.1))
        model.add(Bidirectional(LSTM(units=120, return_sequences=True, recurrent_dropout=0.1)))
        model.add(TimeDistributed(Dense(self.num_entities)))
        model.add(Activation('softmax'))

        plot_model(model, to_file='../models/ner_' + str(datetime.utcnow().microsecond) + '.png')
        print(model.summary())

        model.compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics=['accuracy'])
        tensorboard_callback = TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=False)

        history = model.fit(self.X_train, np.array(self.Y_train), batch_size=32, epochs=5, validation_split=0.2,
                            callbacks=[tensorboard_callback])

        model.save("../models/ner_" + str(datetime.utcnow().microsecond))

        # TODO wtf?
        test_eval = model.evaluate(self.X_test, np.array(self.Y_test))
        print('Test loss:', test_eval[0])
        print('Test accuracy:', test_eval[1])

        return model, history
