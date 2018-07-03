import numpy as np
from datetime import datetime
from keras.models import Model, Input
from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional


class NeuralNetwork(object):

    def __init__(self, num_words, num_entities, X_train, Y_train, X_validation, Y_validation):
        self.num_words = num_words
        self.num_entities = num_entities
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_validation = X_validation
        self.Y_validation = Y_validation

    def train(self):
        input = Input(shape=(120,))
        model = Embedding(input_dim=self.num_words, output_dim=50, input_length=120)(input)
        model = Dropout(0.1)(model)
        model = Bidirectional(LSTM(units=100, return_sequences=True, recurrent_dropout=0.1))(model)
        out = TimeDistributed(Dense(self.num_entities, activation="softmax"))(model)

        model = Model(input, out)

        model.compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"])

        history = model.fit(x=self.X_train, y=np.array(self.Y_train), batch_size=32, epochs=5,
                            validation_data=(self.X_validation, self.Y_validation))

        model.save("../models/ner_" + str(datetime.utcnow().microsecond))

        return model, history
