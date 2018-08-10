import json
import os
from datetime import datetime
from collections import namedtuple
from train import train


def _json_object_hook(d):
    return namedtuple('args', d.keys())(*d.values())


def json2obj(data):
    return json.loads(data, object_hook=_json_object_hook)


def main():
    with open('model_args.txt') as fd:
        data = fd.read()
    args = json2obj(data)
    print(args.rnn_type)

    train(args)


if __name__ == '__main__':
    main()
