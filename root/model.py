import numpy as np
from keras import Input, Model
from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional, concatenate
from keras.utils.vis_utils import plot_model
from keras.callbacks import TensorBoard

from root.constants import MAX_LEN


class NeuralNetwork(object):

    def __init__(self, num_words, num_entities, num_pos, num_chunk, train, test, validation):
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
        # txt_input = Input(shape=(MAX_LEN,), name='txt_input')
        # txt_embed = Embedding(input_dim=self.num_words, output_dim=MAX_LEN, input_length=MAX_LEN, name='txt_embedding',
        #                       weights=([embedding]), trainable=False, mask_zero=True)(txt_input)
        # txt_drpot = Dropout(0.1, name='txt_dropout')(txt_embed)
        #
        # model = Bidirectional(LSTM(units=100, return_sequences=True, recurrent_dropout=0.1))(txt_drpot)
        # model = Bidirectional(LSTM(units=100, return_sequences=True, recurrent_dropout=0.1))(model)
        # model = Bidirectional(LSTM(units=100, return_sequences=True, recurrent_dropout=0.1))(model)
        # out = TimeDistributed(Dense(self.num_entities, activation="softmax"))(model)
        #
        # model = Model(txt_input, out)
        #
        # model.compile(optimizer="rmsprop", loss='categorical_crossentropy', metrics=['accuracy'])

        txt_input = Input(shape=(MAX_LEN,), name='txt_input')
        txt_embed = Embedding(input_dim=self.num_words, output_dim=MAX_LEN, input_length=MAX_LEN, name='txt_embedding', mask_zero=True,
                              trainable=False, weights=([embedding]))(txt_input)
        txt_drpot = Dropout(0.1, name='txt_dropout')(txt_embed)

        pos_input = Input(shape=(MAX_LEN,), name='pos_input')
        pos_embed = Embedding(input_dim=self.num_pos, output_dim=MAX_LEN, input_length=MAX_LEN, name='pos_embedding', mask_zero=True)(
            pos_input)
        pos_drpot = Dropout(0.1, name='pos_dropout')(pos_embed)

        chunk_input = Input(shape=(MAX_LEN,), name='chunk_input')
        chunk_embed = Embedding(input_dim=self.num_chunk, output_dim=MAX_LEN, input_length=MAX_LEN, name='chunk_embedding', mask_zero=True)(
            chunk_input)
        chunk_drpot = Dropout(0.1, name='chunk_dropout')(chunk_embed)

        words_pos = concatenate([txt_drpot, pos_drpot, chunk_drpot], axis=2)

        model = Bidirectional(LSTM(units=100, return_sequences=True, recurrent_dropout=0.1))(words_pos)
        model = Bidirectional(LSTM(units=100, return_sequences=True, recurrent_dropout=0.1))(model)
        out = TimeDistributed(Dense(self.num_entities, activation="softmax"))(model)

        model = Model(inputs=[txt_input, pos_input, chunk_input], outputs=[out])

        model.compile(optimizer="rmsprop", loss='categorical_crossentropy', metrics=['accuracy'])

        plot_model(model, to_file='../models/ner_model_image.png')
        print(model.summary())

        model.compile(optimizer="rmsprop", metrics=['accuracy'], loss='categorical_crossentropy')
        tensorboard_callback = TensorBoard(log_dir='../results/logs', histogram_freq=0, write_graph=True, write_images=True)

        history = model.fit([self.X_train, self.train_pos, self.train_chunk], np.array(self.Y_train), batch_size=32, epochs=epochs,
                            validation_data=([self.X_validation, self.valid_pos, self.valid_chunk], np.array(self.Y_validation)),
                            callbacks=[tensorboard_callback], verbose=1)

        model.save("../models/ner_model")

        test_eval = model.evaluate([self.X_test, self.test_pos, self.test_chunk], np.array(self.Y_test))
        print('Test loss:', test_eval[0])
        print('Test accuracy:', test_eval[1])

        return model, history
