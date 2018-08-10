import numpy as np
import matplotlib.pyplot as plt
import argparse
import re
import os
from keras.callbacks import TensorBoard
from dataset.api import load_dataset
from test_model import test_model
from keras.utils.vis_utils import plot_model
from datetime import datetime
from inputs import inputs_factory
from utils.serialization import save_object
from model import model_factory


def parse_args():
    parser = argparse.ArgumentParser(description='Script for training NER model.')
    parser.add_argument('--max-epochs', type=int, default=1, help='Max number of epochs model will be trained.')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size.')
    parser.add_argument('--train-embeddings', action='store_true', default=False,
                        help='Should gradients be propagated to word embeddings.')
    parser.add_argument('--save-path', default='models',
                        help='Folder where models (and other configs) will be saved during training.')
    parser.add_argument('--inputs', choices=['words', 'words-pos', 'words-chars', 'words-pos-chars'], default='words',
                        help='Inputs to model')
    parser.add_argument('--embeddings-trainable', action='store_true', default=False,
                        help='Whether to train word embeddings.')
    parser.add_argument('--embeddings-type', choices=['glove', 'random'], default='glove',
                        help='Which word embeddings will be used.')
    parser.add_argument('--rnn-type', choices=['LSTM', 'GRU'], default='LSTM', help='Type of RNN cell used in model.')
    parser.add_argument('--rnn-num-layers', type=int, default=1, help='RNN number of layers.')
    parser.add_argument('--rnn-bidirectional', action='store_true', default=False, help='Whether RNN is bidirectional.')
    parser.add_argument('--rnn-hidden-size', type=int, default=100, help='RNN hidden size (number of units).')
    parser.add_argument('--rnn-dropout', type=float, default=0.2, help='RNN dropout probability.')

    args = parser.parse_args()

    # add timestamp to save path
    args.save_path = args.save_path if args.save_path[-1] == os.path.sep else args.save_path + os.path.sep
    args.save_path = args.save_path + datetime.now().strftime("%Y-%m-%d-%H:%M") + os.path.sep

    return args


def create_dir():
    runs = ([x[0] for x in os.walk("results/logs")])
    runs = [x for x in runs if "run" in x]
    runs = list(map(int, re.findall(r'\d+', "".join(runs))))
    runs.sort()
    if len(runs) == 0:
        return "results/logs/run1"

    dir_idx = runs[-1] + 1

    dir = "results/logs/run" + str(dir_idx)

    if not os.path.exists(dir):
        os.makedirs(dir)
        return dir
    else:
        raise FileExistsError('Clear logs dir.')


def plot_train_and_save(history):
    # Plot accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc='lower right')
    plt.savefig('results/model_accuracy.png', dpi=200, format='png', bbox_inches='tight')
    plt.close()

    # Plot loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc='upper right')
    plt.savefig('results/model_loss.png', dpi=200, format='png', bbox_inches='tight')
    plt.close()


def filter_inputs(args, datasets):
    train = []
    val = []
    test = []

    if 'words' in args.inputs:
        train.append(datasets.train.X)
        val.append(datasets.val.X)
        test.append(datasets.test.X)

    if 'pos' in args.inputs:
        train.append(datasets.train.pos)
        val.append(datasets.val.pos)
        test.append(datasets.test.pos)

    if 'chars' in args.inputs:
        train.append(datasets.train.chars)
        val.append(datasets.val.chars)
        test.append(datasets.test.chars)

    return train, val, test


def train(args):
    vocabs, datasets = load_dataset()
    inputs, model_input = inputs_factory(args, vocabs)
    model = model_factory(args, len(vocabs.labels.itos), model_input, inputs)

    # save vocabularies
    save_object(vocabs, args.save_path + 'vocabs')

    # prepare model
    model.compile(optimizer="rmsprop", loss='categorical_crossentropy', metrics=['accuracy'])
    plot_model(model, to_file=args.save_path + 'ner_model_image.png')
    print(model.summary())

    dir = create_dir()

    tensorboard_callback = TensorBoard(log_dir=dir, histogram_freq=0, write_graph=True, write_images=True)

    # get inputs based on args.inputs argument
    train, val, test = filter_inputs(args, datasets)

    history = model.fit(
        train,
        np.array(datasets.train.y), batch_size=args.batch_size, epochs=args.max_epochs,
        validation_data=(val, np.array(datasets.val.y)),
        callbacks=[tensorboard_callback], verbose=1)

    model.save(args.save_path + 'ner_model')

    test_eval = model.evaluate(test, np.array(datasets.test.y))
    print('Test loss:', test_eval[0])
    print('Test accuracy:', test_eval[1])

    test_model(args.save_path, datasets.test, test, vocabs.labels)
    plot_train_and_save(history)


if __name__ == '__main__':
    train(parse_args())
