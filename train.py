import argparse
import datetime
import os

import keras
import keras.callbacks
import keras.utils
import numpy as np
import tensorflow as tf

import dataset
from model import SegNet


def main():

    # Parse arguments.
    parser = argparse.ArgumentParser()
    kwargs = {
        'type': int,
        'default': 100,
        'help': 'The number of times of learning. default: 100'
    }
    parser.add_argument('-e', '--epochs', **kwargs)
    kwargs = {
        'type': int,
        'default': 10,
        'help': 'The frequency of saving model. default: 10'
    }
    parser.add_argument('-c', '--checkpoint_interval', **kwargs)
    kwargs = {
        'type': int,
        'default': 1,
        'help': 'The number of samples contained per mini batch. default: 1'
    }
    parser.add_argument('-b', '--batch_size', **kwargs)
    kwargs = {
        'default': False,
        'action': 'store_true',
        'help': 'Whether store all data to GPU. If not specified this option, use both CPU memory and GPU memory.'
    }
    parser.add_argument('--onmemory', **kwargs)
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
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    directory = f'./logs/{timestamp}/'
    os.makedirs(directory, exist_ok=True)
    tensorboard = keras.callbacks.TensorBoard(
        log_dir=directory,
        histogram_freq=1,
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

    model.save_weights(f'{directory}{filename}'.format(epoch=0))

    if args.onmemory:
        model.fit(
            x=train_x,
            y=train_y,
            validation_data=(test_x, test_y),
            epochs=args.epochs,
            batch_size=args.batch_size,
            class_weight='balanced',
            shuffle=True,
            verbose=1,
            callbacks=[tensorboard, checkpoint])
    else:
        class Generator(keras.utils.Sequence):
            def __init__(self, x, y, batch_size, shuffle):
                self.x = x
                self.y = y
                self.batch_size = batch_size
                self.indices = np.arange(len(self.x))
                self.shuffle = shuffle
                assert len(self.x) == len(self.y)
                assert len(self.x) % self.batch_size == 0

            def __getitem__(self, index):
                i = index * self.batch_size
                indices = range(i, i + self.batch_size)
                # indices = self.indices[i:i + self.batch_size]
                x = self.x[indices]
                y = self.y[indices]
                return x, y

            def __len__(self):
                return len(self.x) // self.batch_size

            def on_epoch_end(self):
                if self.shuffle:
                    self.indices = np.random.permutation(self.indices)

        model.fit_generator(
            generator=Generator(train_x, train_y, args.batch_size, True),
            validation_data=Generator(test_x, test_y, args.batch_size, False),
            epochs=args.epochs,
            class_weight='balanced',
            shuffle=True,
            verbose=1,
            callbacks=[tensorboard, checkpoint])


if __name__ == '__main__':
    main()
