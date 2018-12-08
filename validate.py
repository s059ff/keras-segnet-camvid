import argparse
import os

import keras
import numpy as np
import tensorflow as tf

import cv2
from model import SegNet


def main():

    # Parse arguments.
    parser = argparse.ArgumentParser()
    kwargs = {
        'type': str,
        'help': 'The model file path.',
        'required': True
    }
    parser.add_argument('-m', '--model', **kwargs)
    kwargs = {
        'type': int,
        'default': 10,
        'help': 'The number of samples to evaluate. default: 10'
    }
    parser.add_argument('-n', '--num', **kwargs)
    kwargs = {
        'type': str,
        'default': 'val',
        'help': 'Type of dataset to use. "val" or "test". default: "val"'
    }
    parser.add_argument('-t', '--type', **kwargs)
    args = parser.parse_args()

    # Prepare training data.
    dataset = np.load('./temp/dataset.npz')
    val_x = dataset[f'{args.type}_x']
    val_y = dataset[f'{args.type}_y']

    if args.num < 0 or len(val_x) < args.num:
        args.num = len(val_x)

    # Prepare tensorflow.
    config = tf.ConfigProto(device_count={'GPU': 0})
    session = tf.Session(config=config)
    keras.backend.tensorflow_backend.set_session(session)

    # Prepare model.
    model = SegNet(shape=(360, 480, 3))
    model.load_weights(args.model)

    # Output results.
    head, tail = os.path.split(args.model)
    filename, ext = os.path.splitext(tail)
    os.makedirs(f'{head}/{filename}/', exist_ok=True)

    for i, x, y, t in zip(range(args.num), val_x, model.predict(val_x[:args.num]), val_y):
        cv2.imwrite(f'{head}/{filename}/{args.type}-{i}-input.png', x * 255)
        cv2.imwrite(f'{head}/{filename}/{args.type}-{i}-prediction.png', y * 255)
        cv2.imwrite(f'{head}/{filename}/{args.type}-{i}-teacher.png', t * 255)


if __name__ == '__main__':
    main()
