import numpy as np
from keras import Input, Model
from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional, concatenate, SpatialDropout1D
from keras.utils.vis_utils import plot_model
from keras.callbacks import TensorBoard

from root.constants import MAX_LEN


class NeuralNetwork(object):

    def __init__(self, num_words, num_entities, num_pos, num_chars, train, test, validation):
        self.num_words = num_words
        self.num_entities = num_entities
        self.num_pos = num_pos
        self.num_chars = num_chars
        self.X_train = train.X
        self.Y_train = train.y
        self.X_validation = validation.X
        self.Y_validation = validation.y
        self.X_test = test.X
        self.Y_test = test.y
        self.train_pos = train.pos
        self.test_pos = test.pos
        self.valid_pos = validation.pos

        self.train_characters = train.characters
        self.test_characters = test.characters
        self.valid_characters = validation.characters

    def train(self, epochs, embedding=None):

        # Embedded Words
        txt_input = Input(shape=(MAX_LEN,), name='txt_input')
        txt_embed = Embedding(input_dim=self.num_words, output_dim=MAX_LEN, input_length=MAX_LEN, name='txt_embedding',
                              trainable=False, weights=([embedding]))(txt_input)
        txt_drpot = Dropout(0.1, name='txt_dropout')(txt_embed)

        # Embedded Part of Speech
        pos_input = Input(shape=(MAX_LEN,), name='pos_input')
        pos_embed = Embedding(input_dim=self.num_pos, output_dim=MAX_LEN, input_length=MAX_LEN, name='pos_embedding')(
            pos_input)
        pos_drpot = Dropout(0.1, name='pos_dropout')(pos_embed)

        # Embedded Characters
        char_in = Input(shape=(MAX_LEN, 10,), name="char_input")
        emb_char = TimeDistributed(Embedding(input_dim=self.num_chars, output_dim=10, input_length=10))(char_in)
        char_enc = TimeDistributed(LSTM(units=20, return_sequences=False, recurrent_dropout=0.5))(emb_char)

        # Concatenate inputs
        x = concatenate([txt_drpot, pos_drpot, char_enc], axis=2)
        x = SpatialDropout1D(0.3)(x)

        # Deep Layers
        model = Bidirectional(LSTM(units=100, return_sequences=True, recurrent_dropout=0.1))(x)
        model = Bidirectional(LSTM(units=100, return_sequences=True, recurrent_dropout=0.1))(model)

        # Output
        out = TimeDistributed(Dense(self.num_entities, activation="softmax"))(model)
        model = Model(inputs=[txt_input, pos_input, char_in], outputs=[out])

        model.compile(optimizer="rmsprop", loss='categorical_crossentropy', metrics=['accuracy'])

        plot_model(model, to_file='../models/ner_model_image.png')
        print(model.summary())

        model.compile(optimizer="rmsprop", metrics=['accuracy'], loss='categorical_crossentropy')
        tensorboard_callback = TensorBoard(log_dir='../results/logs', histogram_freq=0, write_graph=True, write_images=True)

        history = model.fit(
            [self.X_train, self.train_pos, np.array(self.train_characters).reshape((len(self.train_characters), 100, 10))],
            np.array(self.Y_train), batch_size=32, epochs=epochs,
            validation_data=(
                [self.X_validation, self.valid_pos, np.array(self.valid_characters).reshape((len(self.valid_characters), 100, 10))],
                np.array(self.Y_validation)),
            callbacks=[tensorboard_callback], verbose=1)

        model.save("../models/ner_model")

        test_eval = model.evaluate(
            [self.X_test, self.test_pos, np.array(self.test_characters).reshape((len(self.test_characters), 100, 10))],
            np.array(self.Y_test))
        print('Test loss:', test_eval[0])
        print('Test accuracy:', test_eval[1])

        return model, history
