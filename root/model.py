import numpy as np
from keras import Sequential
from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional, Activation, GRU
from keras.utils.vis_utils import plot_model
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

    def train(self, epochs, embedding=None):
        model = Sequential()
        if embedding is None:
            model.add(Embedding(input_dim=self.num_words, output_dim=100, mask_zero=True))
        else:
            model.add(Embedding(input_dim=self.num_words, output_dim=100, weights=[embedding], mask_zero=True,
                                trainable=True))

        model.add(Dropout(0.1))
        # model.add(LSTM(units=100, dropout=0.1, return_sequences=True, recurrent_dropout=0.1, use_bias=True))
        # model.add(GRU(units=120, dropout=0.1, return_sequences=True))
        model.add(Bidirectional(LSTM(units=100, return_sequences=True, recurrent_dropout=0.1)))
        # model.add(
        #     Bidirectional(GRU(units=100, dropout=0.1, return_sequences=True, recurrent_dropout=0.1, use_bias=True)))
        model.add(TimeDistributed(Dense(self.num_entities)))
        model.add(Activation('softmax'))

        plot_model(model, to_file='../models/ner_model_image.png')
        print(model.summary())

        model.compile(optimizer="rmsprop", metrics=['accuracy'], loss='categorical_crossentropy')
        tensorboard_callback = TensorBoard(log_dir='../results/logs', histogram_freq=0, write_graph=True,
                                           write_images=True)

        history = model.fit(self.X_train, np.array(self.Y_train), batch_size=32, epochs=epochs,
                            validation_data=(self.X_validation, np.array(self.Y_validation)),
                            callbacks=[tensorboard_callback])

        model.save("../models/ner_model")

        test_eval = model.evaluate(self.X_test, np.array(self.Y_test))
        print('Test loss:', test_eval[0])
        print('Test accuracy:', test_eval[1])

        return model, history
