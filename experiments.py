import json
import os
from collections import namedtuple

from train import train


def _json_object_hook(d):
    return namedtuple('args', d.keys())(*d.values())


def json2obj(data):
    return json.loads(data, object_hook=_json_object_hook)


def main():
    with open('model_args.json') as fd:
        data = fd.read()
    experiment_configs = json2obj(data)

    for args in experiment_configs:
        print(args.rnn_type)
        os.makedirs(args.save_path + 'images', exist_ok=True)
        train(args)


if __name__ == '__main__':
    main()
