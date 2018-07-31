import numpy as np
from keras import Sequential, Input, Model
from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional, Activation, GRU
from keras.utils.vis_utils import plot_model
from keras.callbacks import TensorBoard

from root.constants import MAX_LEN


class NeuralNetwork(object):

    def __init__(self, save_path, num_words, num_entities, num_pos, num_chunk, train, test, validation):
        self.save_path = save_path
        self.num_words = num_words
        self.num_entities = num_entities
        self.num_pos = num_pos
        self.num_chunk = num_chunk
        self.X_train = train.X
        self.Y_train = train.y
        self.X_validation = validation.X
        self.Y_validation = validation.y
        self.X_test = test.X
        self.Y_test = test.y
        self.train_pos = train.pos
        self.test_pos = test.pos
        self.valid_pos = validation.pos

        self.train_chunk = train.chunk
        self.test_chunk = test.chunk
        self.valid_chunk = validation.chunk

    def train(self, epochs, embedding=None):
        txt_input = Input(shape=(None,), name='txt_input')
        txt_embed = Embedding(input_dim=self.num_words, output_dim=MAX_LEN, input_length=None, name='txt_embedding',
                              weights=([embedding]), trainable=False, mask_zero=True)(txt_input)
        txt_drpot = Dropout(0.1, name='txt_dropout')(txt_embed)

        model = Bidirectional(LSTM(units=100, return_sequences=True, recurrent_dropout=0.1))(txt_drpot)
        model = Bidirectional(LSTM(units=100, return_sequences=True, recurrent_dropout=0.1))(model)
        model = Bidirectional(LSTM(units=100, return_sequences=True, recurrent_dropout=0.1))(model)
        out = TimeDistributed(Dense(self.num_entities, activation="softmax"))(model)

        model = Model(txt_input, out)

        model.compile(optimizer="rmsprop", loss='categorical_crossentropy', metrics=['accuracy'])

        plot_model(model, to_file='../models/ner_model_image.png')
        print(model.summary())

        model.compile(optimizer="rmsprop", metrics=['accuracy'], loss='categorical_crossentropy')
        tensorboard_callback = TensorBoard(log_dir='../results/logs', histogram_freq=0, write_graph=True, write_images=True)

        history = model.fit(self.X_train, np.array(self.Y_train), batch_size=32, epochs=epochs,
                            validation_data=(self.X_validation, np.array(self.Y_validation)), callbacks=[tensorboard_callback])

        model.save(self.save_path + 'ner_model')

        test_eval = model.evaluate(self.X_test, np.array(self.Y_test))
        print('Test loss:', test_eval[0])
        print('Test accuracy:', test_eval[1])

        return model, history
