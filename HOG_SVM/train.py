import cv2, glob, os, joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from skimage.feature import hog
from PIL import Image
from sklearn import svm, metrics


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
        else:
            img = cv2.imread(filename)

        # Convert to grayscale and resize img to 64x128
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (64, 128))

        # Calculate Histogram of Oriented Gradient (HOG)
        img_hog = hog(img, orientations=9, pixels_per_cell=(8, 8), visualize=False, cells_per_block=(3, 3))
        X.append(img_hog)
        y.append(cls)


def split_data(X, y, test_size=0.2):
    X = np.float32(X)
    y = np.array(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    print('Train input:', X_train.shape)
    print('Train label:', y_train.shape)

    return X_train, X_test, y_train, y_test


def main():
    global X, y

    # Load features from Train set
    load_features(neg_im, 0)
    load_features(pos_im, 1)

    # Split original Train set into Train and Test
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.3)

    # Build model
    svm_clf = svm.LinearSVC()
    svm_clf.fit(X_train, y_train)

    # Confusion matrix
    y_pred = svm_clf.predict(X_test)
    print(f'Classification report for classifier {svm_clf}')
    print(f'{metrics.classification_report(y_test, y_pred)}')

    # Save model
    joblib.dump(svm_clf, "./models/LinearSVC.dat")
    print('Model saved!')


if __name__ == '__main__':
    main()