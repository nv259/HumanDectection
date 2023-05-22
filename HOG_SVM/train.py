import cv2
import glob
import os
import numpy as np
import matplotlib.pyplot as plt
import argparse
from skimage.feature import hog
from PIL import Image


# Init training set
X = []
y = []

# Environment
pos_im = './images/Train/pos'
neg_im = './images/Train/neg'


def load_features(path, cls):
    for filename in glob.glob(os.path.join(path, '*')):
        if cls == 1:
            # Read image using PIL then convert to cv2
            img = np.array(Image.open(filename).convert('RGB'))
            img = img[:, :, ::-1].copy()

            cv2.imshow('pos', img)
            cv2.waitKey(0)
        else:
            img = cv2.imread(filename)

        # Convert to Gray Scale
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (64, 128))

        # Calculate Histogram of Oriented Gradient (HOG)
        img_hog = hog(img, orientations=9, pixels_per_cell=(8, 8), visualize=False, cells_per_block=(3, 3))
        X.append(img_hog)
        y.append(cls)


def main():
    load_features(neg_im, 0)
    load_features(pos_im, 1)


if __name__ == '__main__':
    main()