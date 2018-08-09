import os
import pickle


def ensure_dir_exists(path):
    if not os.path.isdir(path):
        os.makedirs(path)


def save_object(obj, path):
    ensure_dir_exists(os.path.dirname(path))
    with open(path, 'wb') as fd:
        pickle.dump(obj, fd)


def load_object(path):
    with open(path, 'rb') as fd:
        obj = pickle.load(fd)
    return obj
