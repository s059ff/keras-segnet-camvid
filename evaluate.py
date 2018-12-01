import argparse
import glob
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
    default = './temp/**/model-*.h5'
    parser.add_argument('-m', '--model', type=str, default=default)
    parser.add_argument('-n', '--num', type=int, default=10)
    args = parser.parse_args()

    if args.model is None:
        print('You need to specify model file path using -m(--model) option.')
        exit()

    # Prepare training data.
    os.makedirs('./temp/', exist_ok=True)
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
    paths = sorted(glob.glob(args.model))

    for path in paths:
        print(path)

        model.load_weights(path)
        head, tail = os.path.split(path)
        filename, ext = os.path.splitext(tail)

        # Save images.
        os.makedirs(f'{head}/{filename}/', exist_ok=True)
        num = args.num

        # for i, x, y, t in zip(range(num), train_x[:num], model.predict(train_x[:num]), train_y[:num]):
        #     z = np.dstack((x, y))
        #     cv2.imwrite(f'{head}/{filename}/train-{i}-input.png', x * 255)
        #     cv2.imwrite(f'{head}/{filename}/train-{i}-prediction.png', y * 255)
        #     cv2.imwrite(
        #         f'{head}/{filename}/train-{i}-prediction+.png', z * 255)
        #     cv2.imwrite(f'{head}/{filename}/train-{i}-teaching.png', t * 255)

        for i, x, y, t in zip(range(num), test_x[:num], model.predict(test_x[:num]), test_y[:num]):
            z = np.dstack((x, y))
            cv2.imwrite(f'{head}/{filename}/test-{i}-input.png', x * 255)
            cv2.imwrite(f'{head}/{filename}/test-{i}-prediction.png', y * 255)
            cv2.imwrite(f'{head}/{filename}/test-{i}-prediction+.png', z * 255)
            cv2.imwrite(f'{head}/{filename}/test-{i}-teaching.png', t * 255)


if __name__ == '__main__':
    main()
