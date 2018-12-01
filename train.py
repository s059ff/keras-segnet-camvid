import argparse
import datetime
import os

import keras
import keras.callbacks
import numpy as np
import tensorflow as tf

import dataset
from model import SegNet


def main():

    # Parse arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--epochs', type=int, default=100)
    parser.add_argument('--checkpoint_interval', type=int, default=10)
    parser.add_argument('--logging_interval', type=int, default=1)
    args = parser.parse_args()

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
    config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
    session = tf.Session(config=config)
    keras.backend.tensorflow_backend.set_session(session)

    # Prepare model.
    model = SegNet()
    model.compile(loss='binary_crossentropy',
                  optimizer='adadelta',
                  metrics=['accuracy'])

    # Training.
    timestamp = datetime.datetime.now().isoformat()
    directory = f'./logs/{timestamp}/'
    os.makedirs(directory, exist_ok=True)
    tensorboard = keras.callbacks.TensorBoard(
        log_dir=directory,
        histogram_freq=args.logging_interval,
        write_graph=True,
        write_images=True)

    filename = 'model-{epoch:04d}.h5'
    directory = f'./temp/{timestamp}/'
    os.makedirs(directory, exist_ok=True)
    checkpoint = keras.callbacks.ModelCheckpoint(
        filepath=f'{directory}{filename}',
        monitor='val_loss',
        verbose=0,
        save_best_only=False,
        save_weights_only=False,
        mode='auto',
        period=args.checkpoint_interval)

    model.fit(
        x=train_x,
        y=train_y,
        batch_size=1,
        epochs=args.epochs,
        verbose=1,
        class_weight='balanced',
        validation_data=(test_x, test_y),
        shuffle=True,
        callbacks=[tensorboard, checkpoint])


if __name__ == '__main__':
    main()
