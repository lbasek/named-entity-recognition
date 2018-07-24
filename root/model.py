import numpy as np
from keras import Input, Model
from keras.callbacks import TensorBoard
from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional, concatenate
from keras.utils.vis_utils import plot_model

from root.constants import MAX_LEN


class NeuralNetwork(object):

    def __init__(self, num_words, num_entities, num_pos, train, test, val):
        self.num_words = num_words
        self.num_entities = num_entities
        self.num_pos = num_pos
        self.X_train = train.X
        self.Y_train = train.y
        self.X_validation = val.X
        self.Y_validation = val.y
        self.X_test = test.X
        self.Y_test = test.y
        self.train_pos = train.pos
        self.test_pos = test.pos
        self.validation_pos = val.pos

    def train(self, epochs, embedding=None):

        txt_input = Input(shape=(MAX_LEN,), name='txt_input')
        txt_embed = Embedding(input_dim=self.num_words, output_dim=MAX_LEN, input_length=MAX_LEN, name='txt_embedding',
                              weights=([embedding]), mask_zero=True)(txt_input)
        txt_drpot = Dropout(0.1, name='txt_dropout')(txt_embed)

        model = Bidirectional(LSTM(units=100, return_sequences=True, recurrent_dropout=0.1))(txt_drpot)
        out = TimeDistributed(Dense(self.num_entities, activation="softmax"))(model)

        model = Model(txt_input, out)

        model.compile(optimizer="rmsprop", loss='categorical_crossentropy', metrics=['accuracy'])

        plot_model(model, to_file='../models/ner_model_image.png')

        print(model.summary())

        tensorboard_callback = TensorBoard(log_dir='../results/logs', histogram_freq=0, write_graph=True,
                                           write_images=True)

        history = model.fit(self.X_train, np.array(self.Y_train), batch_size=32, epochs=epochs,
                            validation_data=(self.X_validation, np.array(self.Y_validation)),
                            callbacks=[tensorboard_callback], verbose=1)

        test_eval = model.evaluate(self.X_test, np.array(self.Y_test))

        model.save("../models/ner_model")
        print('Test loss:', test_eval[0])
        print('Test accuracy:', test_eval[1])

        return model, history

    def train_with_features(self, epochs):
        txt_input = Input(shape=(MAX_LEN,), name='txt_input')
        txt_embed = Embedding(input_dim=self.num_words, output_dim=MAX_LEN, input_length=MAX_LEN, name='txt_embedding',
                              mask_zero=True)(txt_input)
        txt_drpot = Dropout(0.1, name='txt_dropout')(txt_embed)

        pos_input = Input(shape=(100,), name='pos_input')
        pos_embed = Embedding(input_dim=self.num_pos, output_dim=100, input_length=100,
                              name='pos_embedding', trainable=True, mask_zero=True)(pos_input)
        pos_drpot = Dropout(0.1, name='pos_dropout')(pos_embed)

        words_pos = concatenate([txt_drpot, pos_drpot], axis=2)

        model = Bidirectional(LSTM(units=200, return_sequences=True, recurrent_dropout=0.1))(words_pos)
        out = TimeDistributed(Dense(self.num_entities, activation="softmax"))(model)

        model = Model(inputs=[txt_input, pos_input], outputs=[out])

        model.compile(optimizer="rmsprop", loss='categorical_crossentropy', metrics=['accuracy'])

        plot_model(model, to_file='../models/ner_model_image.png')
        print(model.summary())

        tensorboard_callback = TensorBoard(log_dir='../results/logs', histogram_freq=0, write_graph=True,
                                           write_images=True)

        history = model.fit([self.X_train, self.train_pos], np.array(self.Y_train), batch_size=32, epochs=epochs,
                            validation_data=([self.X_validation, self.validation_pos], np.array(self.Y_validation)),
                            callbacks=[tensorboard_callback], verbose=1)

        test_eval = model.evaluate([self.X_test, self.test_pos], np.array(self.Y_test))

        model.save("../models/ner_model")
        print('Test loss:', test_eval[0])
        print('Test accuracy:', test_eval[1])

        return model, history
