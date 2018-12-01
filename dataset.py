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
        image[:, :, 0] = cv2.equalizeHist(image[:, :, 0])
        image[:, :, 1] = cv2.equalizeHist(image[:, :, 1])
        image[:, :, 2] = cv2.equalizeHist(image[:, :, 2])
        image = image.astype(np.float) / 255.
        originals.append(image)

        image = cv2.imread(path2)[:, :, 0]
        # '8' means CAR class label.
        annotation = np.where(image == 8, 1, 0)
        annotation = np.reshape(annotation, (*annotation.shape, 1))
        annotations.append(annotation)

    originals = np.array(originals, dtype=np.float)
    annotations = np.array(annotations, dtype=np.float)

    # For debug.
    # N = 3
    # for i, x in zip(range(N), originals[:N]):
    #     cv2.imwrite(f'./temp/example-input-{i}.png', x * 255)
    # for i, y in zip(range(N), annotations[:N]):
    #     cv2.imwrite(f'./temp/example-annotation-{i}.png', y * 255)

    return (originals, annotations)
