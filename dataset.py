import glob
import os

import numpy as np

import cv2


def load(folder='train'):

    # Load dataset.
    originals = []
    annotations = []
    for filename in map(lambda path: os.path.basename(path), glob.glob(f'./dataset/{folder}/*.png')):
        path1 = f'./dataset/{folder}/' + filename
        path2 = f'./dataset/{folder}annot/' + filename

        if not os.path.exists(path1):
            raise Exception(f'{path1} is not found.')
        if not os.path.exists(path2):
            raise Exception(f'{path2} is not found.')

        image = cv2.imread(path1)
        b, g, r = cv2.split(image)
        b = cv2.equalizeHist(b)
        g = cv2.equalizeHist(g)
        r = cv2.equalizeHist(r)
        image = np.dstack((b, g, r))
        image = np.float32(image) / 255.
        originals.append(image)

        image = cv2.imread(path2)[:, :, 0]
        # '8' means CAR class label.
        annotation = np.where(image == 8, 1, 0)
        annotation = np.int8(annotation)
        annotation = np.reshape(annotation, (*annotation.shape, 1))
        annotations.append(annotation)

    originals = np.array(originals, dtype=np.float32)
    annotations = np.array(annotations, dtype=np.int8)

    return (originals, annotations)


def main():
    os.makedirs('./temp/', exist_ok=True)

    train_x, train_y = load(folder='train')
    val_x, val_y = load(folder='val')
    test_x, test_y = load(folder='test')
    data = {
        'train_x': train_x,
        'train_y': train_y,
        'val_x': val_x,
        'val_y': val_y,
        'test_x': test_x,
        'test_y': test_y,
    }
    np.savez_compressed('./temp/dataset.npz', **data)


if __name__ == '__main__':
    main()
