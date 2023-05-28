import cv2, glob, os, joblib, argparse
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
train_pos_im = './images/Train/pos'
train_neg_im = './images/Train/neg'
# test_pos_im = './images/Test/pos'
test_neg_im = './images/Test/neg'
test_pos_im = 'D:\\repo\\HumanDectection\\HOG_SVM\\Datasets\\PRW-v16.04.20\\PRW-v16.04.20\\query_box'


def load_features(path, cls):
    for filename in glob.glob(os.path.join(path, '*')):
        img = cv2.imread(filename)
            
        if img is None:
            # Some file can't be read by cv2, use PIL instead
            img = np.array(Image.open(filename).convert('RGB'))
            img = img[:, :, ::-1].copy()
            
        # print(filename)
        # print(img.shape)
        
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
    # load_features(train_neg_im, 0)
    # load_features(train_pos_im, 1)
    # X_train = np.float32(X)
    # y_train = np.array(y)
    
    X = []
    y = [] 
    
    load_features(test_neg_im, 0) 
    load_features(test_pos_im, 1)
    X_test = np.float32(X)
    y_test = np.array(y)

    # Split original Train set into Train and Test
    # X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.3)

    # Build model
    # svm_clf = svm.LinearSVC()
    # svm_clf.fit(X_train, y_train)
    svm_clf = joblib.load('./models/LinearSVC.dat')
    # Confusion matrix
    y_pred = svm_clf.predict(X_test)
    print(f'Classification report for classifier {svm_clf}')
    print(f'{metrics.classification_report(y_test, y_pred)}')

    # Save model
    # joblib.dump(svm_clf, "./models/LinearSVC.dat")
    print('Model saved!')


if __name__ == '__main__':
    main()
