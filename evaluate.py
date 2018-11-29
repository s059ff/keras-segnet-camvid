import argparse
import os

import keras
import numpy as np
import tensorflow as tf

import cv2
import dataset
from model import SegNet


def main():

    # Parse arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', type=str)
    args = parser.parse_args()

    if args.model is None:
        print('You need to specify model file path using -m(--model) option.')
        exit()

    if not os.path.exists(args.model):
        print(f'Specified model({args.model}) was not found on file system.')
        exit()

    # Prepare training data.
    if not os.path.exists('./temp/train_x.npy') or not os.path.exists('./temp/train_y.npy'):
        train_x, train_y = dataset.load(folder='train')
        np.save('./temp/train_x.npy', train_x)
        np.save('./temp/train_y.npy', train_y)
    else:
        train_x = np.load('./temp/train_x.npy')
        train_y = np.load('./temp/train_y.npy')

    if not os.path.exists('./temp/test_x.npy') or not os.path.exists('./temp/test_y.npy'):
        test_x, test_y = dataset.load(folder='test')
        np.save('./temp/test_x.npy', test_x)
        np.save('./temp/test_y.npy', test_y)
    else:
        test_x = np.load('./temp/test_x.npy')
        test_y = np.load('./temp/test_y.npy')

    # Prepare tensorflow.
    config = tf.ConfigProto(device_count={'GPU': 0})
    session = tf.Session(config=config)
    keras.backend.tensorflow_backend.set_session(session)

    # Prepare model.
    model = SegNet()
    model.load_weights(args.model)

    # Save images.
    N = 100
    for i, x, y, z in zip(range(N), train_x[:N], model.predict(train_x[:N]), train_y[:N]):
        cv2.imwrite(f'./temp/train-{i}-x.png', x * 255)
        cv2.imwrite(f'./temp/train-{i}-y.png', y[:, :, 1] * 255)
        cv2.imwrite(f'./temp/train-{i}-z.png', z[:, :, 1] * 255)

    for i, x, y, z in zip(range(N), test_x[:N], model.predict(test_x[:N]), test_y[:N]):
        cv2.imwrite(f'./temp/test-{i}-x.png', x * 255)
        cv2.imwrite(f'./temp/test-{i}-y.png', y[:, :, 1] * 255)
        cv2.imwrite(f'./temp/test-{i}-z.png', z[:, :, 1] * 255)


if __name__ == '__main__':
    main()
